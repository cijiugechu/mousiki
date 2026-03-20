use mousiki::opus_decoder::{opus_decode, opus_decoder_create};
use mousiki::opus_encoder::{
    OpusEncoderCtlRequest, opus_encode, opus_encoder_create, opus_encoder_ctl,
};
use std::env;
use std::fmt;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAGIC: [u8; 8] = *b"OPUSBEN1";
const MAX_PACKET_SIZE: usize = 3 * 1276;

#[cfg(feature = "dhat_alloc")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat_alloc")]
    let _profiler = dhat::Profiler::new_heap();

    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), BenchError> {
    let mut args = env::args();
    let _program = args.next();
    let command = args.next().ok_or(BenchError::Usage(usage()))?;
    let rest: Vec<String> = args.collect();

    match command.as_str() {
        "packets" => {
            let config = PacketCorpusArgs::parse(&rest)?;
            write_packet_corpus(&config)?;
        }
        "encode" => {
            let config = EncodeBenchArgs::parse(&rest)?;
            let stats = benchmark_encode(&config)?;
            print_result(&stats, config.output);
        }
        "decode" => {
            let config = DecodeBenchArgs::parse(&rest)?;
            let stats = benchmark_decode(&config)?;
            print_result(&stats, config.output);
        }
        _ => return Err(BenchError::Usage(usage())),
    }

    Ok(())
}

fn usage() -> String {
    [
        "usage:",
        "  codec_bench packets --input INPUT.pcm --output OUTPUT.opusbench \\",
        "    --sample-rate 48000 --channels 2 --frame-size 960 \\",
        "    --application audio --bitrate 64000 [--complexity 10] [--bitrate-mode cvbr] [--max-frames N]",
        "  codec_bench encode --input INPUT.pcm --sample-rate 48000 --channels 2 --frame-size 960 \\",
        "    --application audio --bitrate 64000 [--complexity 10] [--bitrate-mode cvbr] \\",
        "    [--warmup 3] [--measure 10] [--max-frames N] [--format text|csv] [--no-header]",
        "  codec_bench decode --packets INPUT.opusbench [--warmup 3] [--measure 10] [--max-frames N] \\",
        "    [--format text|csv] [--no-header]",
    ]
    .join("\n")
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Text,
    Csv,
}

#[derive(Clone, Copy)]
struct OutputOptions {
    format: OutputFormat,
    header: bool,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            format: OutputFormat::Text,
            header: true,
        }
    }
}

#[derive(Clone, Copy)]
enum BitrateMode {
    Vbr,
    Cvbr,
    Cbr,
}

impl BitrateMode {
    fn parse(value: &str) -> Result<Self, BenchError> {
        match value {
            "vbr" => Ok(Self::Vbr),
            "cvbr" => Ok(Self::Cvbr),
            "cbr" => Ok(Self::Cbr),
            _ => Err(BenchError::Argument(format!(
                "invalid --bitrate-mode '{value}', expected vbr|cvbr|cbr"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Vbr => "vbr",
            Self::Cvbr => "cvbr",
            Self::Cbr => "cbr",
        }
    }
}

#[derive(Clone, Copy)]
enum ApplicationArg {
    Voip,
    Audio,
    RestrictedLowDelay,
}

impl ApplicationArg {
    fn parse(value: &str) -> Result<Self, BenchError> {
        match value {
            "voip" => Ok(Self::Voip),
            "audio" => Ok(Self::Audio),
            "restricted-lowdelay" | "restricted_lowdelay" | "lowdelay" => {
                Ok(Self::RestrictedLowDelay)
            }
            _ => Err(BenchError::Argument(format!(
                "invalid --application '{value}', expected voip|audio|restricted-lowdelay"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Voip => "voip",
            Self::Audio => "audio",
            Self::RestrictedLowDelay => "restricted-lowdelay",
        }
    }

    const fn to_opus_int(self) -> i32 {
        match self {
            Self::Voip => 2048,
            Self::Audio => 2049,
            Self::RestrictedLowDelay => 2051,
        }
    }
}

#[derive(Clone, Copy)]
struct EncodeConfig {
    sample_rate: i32,
    channels: i32,
    frame_size: usize,
    application: ApplicationArg,
    bitrate: i32,
    complexity: i32,
    bitrate_mode: BitrateMode,
}

impl EncodeConfig {
    const fn max_frame_size(self) -> usize {
        self.frame_size * 6
    }
}

struct PacketCorpusArgs {
    input: PathBuf,
    output: PathBuf,
    encode: EncodeConfig,
    max_frames: Option<usize>,
}

impl PacketCorpusArgs {
    fn parse(args: &[String]) -> Result<Self, BenchError> {
        let mut input = None;
        let mut output = None;
        let mut config = EncodeDefaults::default();
        let mut max_frames = None;

        let mut idx = 0;
        while idx < args.len() {
            match args[idx].as_str() {
                "--input" => input = Some(PathBuf::from(next_value(args, &mut idx, "--input")?)),
                "--output" => output = Some(PathBuf::from(next_value(args, &mut idx, "--output")?)),
                "--sample-rate" => config.sample_rate = parse_i32(args, &mut idx, "--sample-rate")?,
                "--channels" => config.channels = parse_i32(args, &mut idx, "--channels")?,
                "--frame-size" => config.frame_size = parse_usize(args, &mut idx, "--frame-size")?,
                "--application" => {
                    config.application =
                        ApplicationArg::parse(next_value(args, &mut idx, "--application")?)?
                }
                "--bitrate" => config.bitrate = parse_i32(args, &mut idx, "--bitrate")?,
                "--complexity" => config.complexity = parse_i32(args, &mut idx, "--complexity")?,
                "--bitrate-mode" => {
                    config.bitrate_mode =
                        BitrateMode::parse(next_value(args, &mut idx, "--bitrate-mode")?)?
                }
                "--max-frames" => max_frames = Some(parse_usize(args, &mut idx, "--max-frames")?),
                other => return Err(BenchError::Argument(format!("unknown argument '{other}'"))),
            }
            idx += 1;
        }

        Ok(Self {
            input: input.ok_or_else(|| BenchError::Argument("missing --input".into()))?,
            output: output.ok_or_else(|| BenchError::Argument("missing --output".into()))?,
            encode: config.finish(),
            max_frames,
        })
    }
}

struct EncodeBenchArgs {
    input: PathBuf,
    encode: EncodeConfig,
    warmup: usize,
    measure: usize,
    max_frames: Option<usize>,
    output: OutputOptions,
}

impl EncodeBenchArgs {
    fn parse(args: &[String]) -> Result<Self, BenchError> {
        let mut input = None;
        let mut config = EncodeDefaults::default();
        let mut warmup = 3;
        let mut measure = 10;
        let mut max_frames = None;
        let mut output = OutputOptions::default();

        let mut idx = 0;
        while idx < args.len() {
            match args[idx].as_str() {
                "--input" => input = Some(PathBuf::from(next_value(args, &mut idx, "--input")?)),
                "--sample-rate" => config.sample_rate = parse_i32(args, &mut idx, "--sample-rate")?,
                "--channels" => config.channels = parse_i32(args, &mut idx, "--channels")?,
                "--frame-size" => config.frame_size = parse_usize(args, &mut idx, "--frame-size")?,
                "--application" => {
                    config.application =
                        ApplicationArg::parse(next_value(args, &mut idx, "--application")?)?
                }
                "--bitrate" => config.bitrate = parse_i32(args, &mut idx, "--bitrate")?,
                "--complexity" => config.complexity = parse_i32(args, &mut idx, "--complexity")?,
                "--bitrate-mode" => {
                    config.bitrate_mode =
                        BitrateMode::parse(next_value(args, &mut idx, "--bitrate-mode")?)?
                }
                "--warmup" => warmup = parse_usize(args, &mut idx, "--warmup")?,
                "--measure" => measure = parse_usize(args, &mut idx, "--measure")?,
                "--max-frames" => max_frames = Some(parse_usize(args, &mut idx, "--max-frames")?),
                "--format" => {
                    output.format = parse_output_format(next_value(args, &mut idx, "--format")?)?
                }
                "--no-header" => output.header = false,
                other => return Err(BenchError::Argument(format!("unknown argument '{other}'"))),
            }
            idx += 1;
        }

        Ok(Self {
            input: input.ok_or_else(|| BenchError::Argument("missing --input".into()))?,
            encode: config.finish(),
            warmup,
            measure,
            max_frames,
            output,
        })
    }
}

struct DecodeBenchArgs {
    packets: PathBuf,
    warmup: usize,
    measure: usize,
    max_frames: Option<usize>,
    output: OutputOptions,
}

impl DecodeBenchArgs {
    fn parse(args: &[String]) -> Result<Self, BenchError> {
        let mut packets = None;
        let mut warmup = 3;
        let mut measure = 10;
        let mut max_frames = None;
        let mut output = OutputOptions::default();

        let mut idx = 0;
        while idx < args.len() {
            match args[idx].as_str() {
                "--packets" => {
                    packets = Some(PathBuf::from(next_value(args, &mut idx, "--packets")?))
                }
                "--warmup" => warmup = parse_usize(args, &mut idx, "--warmup")?,
                "--measure" => measure = parse_usize(args, &mut idx, "--measure")?,
                "--max-frames" => max_frames = Some(parse_usize(args, &mut idx, "--max-frames")?),
                "--format" => {
                    output.format = parse_output_format(next_value(args, &mut idx, "--format")?)?
                }
                "--no-header" => output.header = false,
                other => return Err(BenchError::Argument(format!("unknown argument '{other}'"))),
            }
            idx += 1;
        }

        Ok(Self {
            packets: packets.ok_or_else(|| BenchError::Argument("missing --packets".into()))?,
            warmup,
            measure,
            max_frames,
            output,
        })
    }
}

struct EncodeDefaults {
    sample_rate: i32,
    channels: i32,
    frame_size: usize,
    application: ApplicationArg,
    bitrate: i32,
    complexity: i32,
    bitrate_mode: BitrateMode,
}

impl Default for EncodeDefaults {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            frame_size: 960,
            application: ApplicationArg::Audio,
            bitrate: 64_000,
            complexity: 10,
            bitrate_mode: BitrateMode::Cvbr,
        }
    }
}

impl EncodeDefaults {
    fn finish(self) -> EncodeConfig {
        EncodeConfig {
            sample_rate: self.sample_rate,
            channels: self.channels,
            frame_size: self.frame_size,
            application: self.application,
            bitrate: self.bitrate,
            complexity: self.complexity,
            bitrate_mode: self.bitrate_mode,
        }
    }
}

fn parse_output_format(value: &str) -> Result<OutputFormat, BenchError> {
    match value {
        "text" => Ok(OutputFormat::Text),
        "csv" => Ok(OutputFormat::Csv),
        _ => Err(BenchError::Argument(format!(
            "invalid --format '{value}', expected text|csv"
        ))),
    }
}

fn next_value<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str, BenchError> {
    *idx += 1;
    args.get(*idx)
        .map(String::as_str)
        .ok_or_else(|| BenchError::Argument(format!("missing value for {flag}")))
}

fn parse_i32(args: &[String], idx: &mut usize, flag: &str) -> Result<i32, BenchError> {
    next_value(args, idx, flag)?
        .parse::<i32>()
        .map_err(|_| BenchError::Argument(format!("invalid integer for {flag}")))
}

fn parse_usize(args: &[String], idx: &mut usize, flag: &str) -> Result<usize, BenchError> {
    next_value(args, idx, flag)?
        .parse::<usize>()
        .map_err(|_| BenchError::Argument(format!("invalid integer for {flag}")))
}

struct PacketCorpus {
    header: PacketHeader,
    packets: Vec<Vec<u8>>,
}

#[derive(Clone, Copy)]
struct PacketHeader {
    sample_rate: u32,
    channels: u16,
    frame_size: u16,
    application: u32,
    bitrate: i32,
    complexity: u8,
    bitrate_mode: BitrateMode,
}

impl PacketHeader {
    fn write(self, mut writer: impl Write) -> Result<(), BenchError> {
        writer.write_all(&MAGIC)?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        writer.write_all(&self.channels.to_le_bytes())?;
        writer.write_all(&self.frame_size.to_le_bytes())?;
        writer.write_all(&self.application.to_le_bytes())?;
        writer.write_all(&self.bitrate.to_le_bytes())?;
        writer.write_all(&[self.complexity])?;
        writer.write_all(&[match self.bitrate_mode {
            BitrateMode::Vbr => 0,
            BitrateMode::Cvbr => 1,
            BitrateMode::Cbr => 2,
        }])?;
        writer.write_all(&[0, 0])?;
        Ok(())
    }

    fn read(mut reader: impl Read) -> Result<Self, BenchError> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(BenchError::Corpus("invalid packet corpus magic".into()));
        }

        let sample_rate = read_u32(&mut reader)?;
        let channels = read_u16(&mut reader)?;
        let frame_size = read_u16(&mut reader)?;
        let application = read_u32(&mut reader)?;
        let bitrate = read_i32(&mut reader)?;
        let complexity = read_u8(&mut reader)?;
        let bitrate_mode = match read_u8(&mut reader)? {
            0 => BitrateMode::Vbr,
            1 => BitrateMode::Cvbr,
            2 => BitrateMode::Cbr,
            value => {
                return Err(BenchError::Corpus(format!(
                    "invalid packet corpus bitrate mode {value}"
                )));
            }
        };
        let _reserved = read_u16(&mut reader)?;

        Ok(Self {
            sample_rate,
            channels,
            frame_size,
            application,
            bitrate,
            complexity,
            bitrate_mode,
        })
    }
}

fn read_u8(reader: &mut impl Read) -> Result<u8, BenchError> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(reader: &mut impl Read) -> Result<u16, BenchError> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(reader: &mut impl Read) -> Result<u32, BenchError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(reader: &mut impl Read) -> Result<i32, BenchError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn write_packet_corpus(config: &PacketCorpusArgs) -> Result<(), BenchError> {
    let pcm = load_pcm_samples(&config.input)?;
    let frames = frame_count(pcm.len(), config.encode)?;
    let frame_limit = config.max_frames.unwrap_or(frames).min(frames);
    if frame_limit == 0 {
        return Err(BenchError::Argument(
            "packet corpus would contain zero frames".into(),
        ));
    }

    let corpus = encode_corpus(&pcm, config.encode, frame_limit)?;
    let output = &config.output;
    let mut file = fs::File::create(output)?;
    corpus.header.write(&mut file)?;
    for packet in &corpus.packets {
        let len = u16::try_from(packet.len())
            .map_err(|_| BenchError::Corpus(format!("packet too large: {}", packet.len())))?;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(packet)?;
    }

    println!(
        "wrote {} packets to {}",
        corpus.packets.len(),
        output.display()
    );
    Ok(())
}

fn encode_corpus(
    pcm: &[i16],
    config: EncodeConfig,
    frame_limit: usize,
) -> Result<PacketCorpus, BenchError> {
    let mut encoder = create_encoder(config)?;
    let channels = usize::try_from(config.channels)
        .map_err(|_| BenchError::Argument("channels out of range".into()))?;
    let frame_samples = config.frame_size * channels;
    let mut packet_buf = vec![0u8; MAX_PACKET_SIZE];
    let mut packets = Vec::with_capacity(frame_limit);

    for frame in 0..frame_limit {
        let start = frame * frame_samples;
        let end = start + frame_samples;
        let len = opus_encode(
            &mut encoder,
            &pcm[start..end],
            config.frame_size,
            &mut packet_buf,
        )
        .map_err(BenchError::Encode)?;
        packets.push(packet_buf[..len].to_vec());
    }

    Ok(PacketCorpus {
        header: PacketHeader {
            sample_rate: u32::try_from(config.sample_rate)
                .map_err(|_| BenchError::Argument("sample rate out of range".into()))?,
            channels: u16::try_from(config.channels)
                .map_err(|_| BenchError::Argument("channels out of range".into()))?,
            frame_size: u16::try_from(config.frame_size)
                .map_err(|_| BenchError::Argument("frame size out of range".into()))?,
            application: u32::try_from(config.application.to_opus_int())
                .map_err(|_| BenchError::Argument("application out of range".into()))?,
            bitrate: config.bitrate,
            complexity: u8::try_from(config.complexity)
                .map_err(|_| BenchError::Argument("complexity out of range".into()))?,
            bitrate_mode: config.bitrate_mode,
        },
        packets,
    })
}

fn benchmark_encode(config: &EncodeBenchArgs) -> Result<BenchResult, BenchError> {
    let pcm = load_pcm_samples(&config.input)?;
    let total_frames = frame_count(pcm.len(), config.encode)?;
    let frame_limit = config.max_frames.unwrap_or(total_frames).min(total_frames);
    if frame_limit == 0 {
        return Err(BenchError::Argument(
            "encode benchmark would process zero frames".into(),
        ));
    }

    let channels = usize::try_from(config.encode.channels)
        .map_err(|_| BenchError::Argument("channels out of range".into()))?;
    let frame_samples = config.encode.frame_size * channels;
    let mut measurements = Vec::with_capacity(config.measure);

    for _ in 0..config.warmup {
        run_encode_iteration(&pcm, config.encode, frame_limit, frame_samples)?;
    }
    for _ in 0..config.measure {
        measurements.push(run_encode_iteration(
            &pcm,
            config.encode,
            frame_limit,
            frame_samples,
        )?);
    }

    Ok(BenchResult::from_measurements(
        "rust",
        "encode",
        config.encode,
        frame_limit,
        config.warmup,
        config.measure,
        measurements,
    ))
}

fn run_encode_iteration(
    pcm: &[i16],
    config: EncodeConfig,
    frame_limit: usize,
    frame_samples: usize,
) -> Result<u128, BenchError> {
    let mut encoder = create_encoder(config)?;
    let mut packet_buf = vec![0u8; MAX_PACKET_SIZE];
    let start = Instant::now();
    for frame in 0..frame_limit {
        let offset = frame * frame_samples;
        let end = offset + frame_samples;
        let _ = opus_encode(
            &mut encoder,
            &pcm[offset..end],
            config.frame_size,
            &mut packet_buf,
        )
        .map_err(BenchError::Encode)?;
    }
    Ok(start.elapsed().as_nanos())
}

fn benchmark_decode(config: &DecodeBenchArgs) -> Result<BenchResult, BenchError> {
    let corpus = load_packet_corpus(&config.packets)?;
    let total_frames = corpus.packets.len();
    let frame_limit = config.max_frames.unwrap_or(total_frames).min(total_frames);
    if frame_limit == 0 {
        return Err(BenchError::Argument(
            "decode benchmark would process zero frames".into(),
        ));
    }

    let encode = EncodeConfig {
        sample_rate: i32::try_from(corpus.header.sample_rate)
            .map_err(|_| BenchError::Corpus("sample rate out of range".into()))?,
        channels: i32::from(corpus.header.channels),
        frame_size: usize::from(corpus.header.frame_size),
        application: match corpus.header.application {
            2048 => ApplicationArg::Voip,
            2049 => ApplicationArg::Audio,
            2051 => ApplicationArg::RestrictedLowDelay,
            value => {
                return Err(BenchError::Corpus(format!(
                    "invalid application code {value}"
                )));
            }
        },
        bitrate: corpus.header.bitrate,
        complexity: i32::from(corpus.header.complexity),
        bitrate_mode: corpus.header.bitrate_mode,
    };

    let max_frame_size = encode.max_frame_size();
    let channels = usize::try_from(encode.channels)
        .map_err(|_| BenchError::Corpus("channels out of range".into()))?;
    let mut measurements = Vec::with_capacity(config.measure);

    for _ in 0..config.warmup {
        run_decode_iteration(&corpus, encode, frame_limit, max_frame_size, channels)?;
    }
    for _ in 0..config.measure {
        measurements.push(run_decode_iteration(
            &corpus,
            encode,
            frame_limit,
            max_frame_size,
            channels,
        )?);
    }

    Ok(BenchResult::from_measurements(
        "rust",
        "decode",
        encode,
        frame_limit,
        config.warmup,
        config.measure,
        measurements,
    ))
}

fn run_decode_iteration(
    corpus: &PacketCorpus,
    config: EncodeConfig,
    frame_limit: usize,
    max_frame_size: usize,
    channels: usize,
) -> Result<u128, BenchError> {
    let mut decoder =
        opus_decoder_create(config.sample_rate, config.channels).map_err(BenchError::DecodeInit)?;
    let mut pcm = vec![0i16; max_frame_size * channels];
    let start = Instant::now();
    for packet in corpus.packets.iter().take(frame_limit) {
        let _ = opus_decode(
            &mut decoder,
            Some(packet),
            packet.len(),
            &mut pcm,
            max_frame_size,
            false,
        )
        .map_err(BenchError::Decode)?;
    }
    Ok(start.elapsed().as_nanos())
}

fn create_encoder(
    config: EncodeConfig,
) -> Result<mousiki::opus_encoder::OpusEncoder<'static>, BenchError> {
    let mut encoder = opus_encoder_create(
        config.sample_rate,
        config.channels,
        config.application.to_opus_int(),
    )
    .map_err(BenchError::EncodeInit)?;

    opus_encoder_ctl(
        &mut encoder,
        OpusEncoderCtlRequest::SetBitrate(config.bitrate),
    )
    .map_err(BenchError::EncodeCtl)?;
    opus_encoder_ctl(
        &mut encoder,
        OpusEncoderCtlRequest::SetComplexity(config.complexity),
    )
    .map_err(BenchError::EncodeCtl)?;
    match config.bitrate_mode {
        BitrateMode::Vbr => {
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbr(true))
                .map_err(BenchError::EncodeCtl)?;
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbrConstraint(false))
                .map_err(BenchError::EncodeCtl)?;
        }
        BitrateMode::Cvbr => {
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbr(true))
                .map_err(BenchError::EncodeCtl)?;
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbrConstraint(true))
                .map_err(BenchError::EncodeCtl)?;
        }
        BitrateMode::Cbr => {
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbr(false))
                .map_err(BenchError::EncodeCtl)?;
            opus_encoder_ctl(&mut encoder, OpusEncoderCtlRequest::SetVbrConstraint(true))
                .map_err(BenchError::EncodeCtl)?;
        }
    }
    Ok(encoder)
}

fn load_pcm_samples(path: &Path) -> Result<Vec<i16>, BenchError> {
    let bytes = fs::read(path)?;
    if !bytes.len().is_multiple_of(2) {
        return Err(BenchError::Input(format!(
            "pcm byte length must be even, got {}",
            bytes.len()
        )));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(samples)
}

fn frame_count(sample_count: usize, config: EncodeConfig) -> Result<usize, BenchError> {
    let channels = usize::try_from(config.channels)
        .map_err(|_| BenchError::Argument("channels out of range".into()))?;
    let frame_samples = config.frame_size * channels;
    if frame_samples == 0 {
        return Err(BenchError::Argument("frame size must be non-zero".into()));
    }
    Ok(sample_count / frame_samples)
}

fn load_packet_corpus(path: &Path) -> Result<PacketCorpus, BenchError> {
    let bytes = fs::read(path)?;
    let mut cursor = Cursor::new(bytes);
    let header = PacketHeader::read(&mut cursor)?;
    let mut packets = Vec::new();

    loop {
        let position = usize::try_from(cursor.position())
            .map_err(|_| BenchError::Corpus("cursor position overflow".into()))?;
        if position == cursor.get_ref().len() {
            break;
        }
        if position + 2 > cursor.get_ref().len() {
            return Err(BenchError::Corpus("truncated packet length".into()));
        }
        let len = usize::from(read_u16(&mut cursor)?);
        let position = usize::try_from(cursor.position())
            .map_err(|_| BenchError::Corpus("cursor position overflow".into()))?;
        let end = position
            .checked_add(len)
            .ok_or_else(|| BenchError::Corpus("packet length overflow".into()))?;
        if end > cursor.get_ref().len() {
            return Err(BenchError::Corpus("truncated packet data".into()));
        }
        packets.push(cursor.get_ref()[position..end].to_vec());
        cursor.set_position(
            u64::try_from(end)
                .map_err(|_| BenchError::Corpus("cursor position overflow".into()))?,
        );
    }

    Ok(PacketCorpus { header, packets })
}

struct BenchResult {
    implementation: &'static str,
    operation: &'static str,
    config: EncodeConfig,
    frames: usize,
    warmup: usize,
    measure: usize,
    median_ns_per_frame: f64,
    p95_ns_per_frame: f64,
    median_packets_per_sec: f64,
    median_realtime_x: f64,
}

impl BenchResult {
    fn from_measurements(
        implementation: &'static str,
        operation: &'static str,
        config: EncodeConfig,
        frames: usize,
        warmup: usize,
        measure: usize,
        measurements: Vec<u128>,
    ) -> Self {
        let mut sorted = measurements;
        sorted.sort_unstable();
        let median_elapsed = percentile(&sorted, 50);
        let p95_elapsed = percentile(&sorted, 95);
        let median_ns_per_frame = median_elapsed as f64 / frames as f64;
        let p95_ns_per_frame = p95_elapsed as f64 / frames as f64;
        let frames_per_sec = 1_000_000_000.0 / median_ns_per_frame;
        let realtime_x = frames_per_sec * config.frame_size as f64 / config.sample_rate as f64;

        Self {
            implementation,
            operation,
            config,
            frames,
            warmup,
            measure,
            median_ns_per_frame,
            p95_ns_per_frame,
            median_packets_per_sec: frames_per_sec,
            median_realtime_x: realtime_x,
        }
    }
}

fn percentile(values: &[u128], percentile: usize) -> u128 {
    debug_assert!(!values.is_empty());
    if values.len() == 1 {
        return values[0];
    }
    let idx = (values.len() - 1) * percentile / 100;
    values[idx]
}

fn print_result(result: &BenchResult, output: OutputOptions) {
    match output.format {
        OutputFormat::Text => print_text(result),
        OutputFormat::Csv => print_csv(result, output.header),
    }
}

fn print_text(result: &BenchResult) {
    println!("implementation={}", result.implementation);
    println!("operation={}", result.operation);
    println!("sample_rate={}", result.config.sample_rate);
    println!("channels={}", result.config.channels);
    println!("frame_size={}", result.config.frame_size);
    println!("application={}", result.config.application.as_str());
    println!("bitrate={}", result.config.bitrate);
    println!("complexity={}", result.config.complexity);
    println!("bitrate_mode={}", result.config.bitrate_mode.as_str());
    println!("frames={}", result.frames);
    println!("warmup_iters={}", result.warmup);
    println!("measure_iters={}", result.measure);
    println!("median_ns_per_frame={:.3}", result.median_ns_per_frame);
    println!("p95_ns_per_frame={:.3}", result.p95_ns_per_frame);
    println!(
        "median_packets_per_sec={:.3}",
        result.median_packets_per_sec
    );
    println!("median_realtime_x={:.3}", result.median_realtime_x);
}

fn print_csv(result: &BenchResult, header: bool) {
    if header {
        println!(
            "implementation,operation,sample_rate,channels,frame_size,application,bitrate,complexity,bitrate_mode,frames,warmup_iters,measure_iters,median_ns_per_frame,p95_ns_per_frame,median_packets_per_sec,median_realtime_x"
        );
    }
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
        result.implementation,
        result.operation,
        result.config.sample_rate,
        result.config.channels,
        result.config.frame_size,
        result.config.application.as_str(),
        result.config.bitrate,
        result.config.complexity,
        result.config.bitrate_mode.as_str(),
        result.frames,
        result.warmup,
        result.measure,
        result.median_ns_per_frame,
        result.p95_ns_per_frame,
        result.median_packets_per_sec,
        result.median_realtime_x,
    );
}

enum BenchError {
    Usage(String),
    Argument(String),
    Input(String),
    Corpus(String),
    Io(std::io::Error),
    EncodeInit(mousiki::opus_encoder::OpusEncoderInitError),
    EncodeCtl(mousiki::opus_encoder::OpusEncoderCtlError),
    Encode(mousiki::opus_encoder::OpusEncodeError),
    DecodeInit(mousiki::opus_decoder::OpusDecoderInitError),
    Decode(mousiki::opus_decoder::OpusDecodeError),
}

impl From<std::io::Error> for BenchError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) => f.write_str(message),
            Self::Argument(message) | Self::Input(message) | Self::Corpus(message) => {
                f.write_str(message)
            }
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::EncodeInit(err) => write!(f, "encoder init failed: {err:?}"),
            Self::EncodeCtl(err) => write!(f, "encoder ctl failed: {err:?}"),
            Self::Encode(err) => write!(f, "encode failed: {err:?}"),
            Self::DecodeInit(err) => write!(f, "decoder init failed: {err:?}"),
            Self::Decode(err) => write!(f, "decode failed: {err:?}"),
        }
    }
}
