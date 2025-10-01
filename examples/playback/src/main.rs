use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig, StreamError};
use mousiki::decoder::{Decoder, DecoderError};
use mousiki::oggreader::{OggRead, OggReader, OggReaderError, ReadError};
use std::env;
use std::fs::File;
use std::io;
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const OPUS_TAGS_SIGNATURE: &[u8] = b"OpusTags";
const PCM_BYTES_PER_FRAME: usize = 1920;
const MAX_SEGMENT_COUNT: usize = 255;
const MAX_SEGMENT_SIZE: usize = 255;

static PLAYBACK_FINISHED: AtomicBool = AtomicBool::new(false);

struct FileStream {
    file: File,
}

impl FileStream {
    fn new(file: File) -> Self {
        Self { file }
    }
}

impl OggRead for FileStream {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        use std::io::Read;

        loop {
            match self.file.read(buf) {
                Ok(n) => return Ok(n),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(0),
                Err(_) => return Err(ReadError::Other),
            }
        }
    }
}

struct Player {
    ogg: OggReader<FileStream>,
    decoder: Decoder,
    pcm: [u8; PCM_BYTES_PER_FRAME],
    pcm_offset: usize,
    pcm_filled: usize,
    segment_storage: [[u8; MAX_SEGMENT_SIZE]; MAX_SEGMENT_COUNT],
    segment_lengths: [usize; MAX_SEGMENT_COUNT],
    total_segments: usize,
    next_segment: usize,
    finished: bool,
}

impl Player {
    fn new(ogg: OggReader<FileStream>) -> Self {
        Self {
            ogg,
            decoder: Decoder::new(),
            pcm: [0u8; PCM_BYTES_PER_FRAME],
            pcm_offset: 0,
            pcm_filled: 0,
            segment_storage: [[0u8; MAX_SEGMENT_SIZE]; MAX_SEGMENT_COUNT],
            segment_lengths: [0usize; MAX_SEGMENT_COUNT],
            total_segments: 0,
            next_segment: 0,
            finished: false,
        }
    }

    fn fill_samples_i16(&mut self, data: &mut [i16], channels: u16) -> Result<bool, ExampleError> {
        let channels = channels as usize;
        if channels == 0 {
            return Ok(true);
        }

        let mut index = 0usize;
        let mut reached_end = false;
        while index + channels <= data.len() {
            match self.next_sample()? {
                Some(sample) => {
                    for channel in 0..channels {
                        data[index + channel] = sample;
                    }
                    index += channels;
                }
                None => {
                    reached_end = true;
                    break;
                }
            }
        }

        if index < data.len() {
            for value in data[index..].iter_mut() {
                *value = 0;
            }
        }

        Ok(reached_end)
    }

    fn fill_samples_f32(&mut self, data: &mut [f32], channels: u16) -> Result<bool, ExampleError> {
        let channels = channels as usize;
        if channels == 0 {
            return Ok(true);
        }

        let mut index = 0usize;
        let mut reached_end = false;
        while index + channels <= data.len() {
            match self.next_sample()? {
                Some(sample) => {
                    let float_sample = (sample as f32) / 32768.0;
                    for channel in 0..channels {
                        data[index + channel] = float_sample;
                    }
                    index += channels;
                }
                None => {
                    reached_end = true;
                    break;
                }
            }
        }

        if index < data.len() {
            for value in data[index..].iter_mut() {
                *value = 0.0;
            }
        }

        Ok(reached_end)
    }

    fn fill_samples_u16(&mut self, data: &mut [u16], channels: u16) -> Result<bool, ExampleError> {
        let channels = channels as usize;
        if channels == 0 {
            return Ok(true);
        }

        let mut index = 0usize;
        let mut reached_end = false;
        while index + channels <= data.len() {
            match self.next_sample()? {
                Some(sample) => {
                    let unsigned = (sample as i32 + 32_768) as u16;
                    for channel in 0..channels {
                        data[index + channel] = unsigned;
                    }
                    index += channels;
                }
                None => {
                    reached_end = true;
                    break;
                }
            }
        }

        if index < data.len() {
            for value in data[index..].iter_mut() {
                *value = 0u16;
            }
        }

        Ok(reached_end)
    }

    fn next_sample(&mut self) -> Result<Option<i16>, ExampleError> {
        if self.finished {
            return Ok(None);
        }

        if self.pcm_offset >= self.pcm_filled {
            if !self.decode_next_segment()? {
                self.finished = true;
                return Ok(None);
            }
        }

        let sample = i16::from_le_bytes([self.pcm[self.pcm_offset], self.pcm[self.pcm_offset + 1]]);
        self.pcm_offset += 2;
        Ok(Some(sample))
    }

    fn decode_next_segment(&mut self) -> Result<bool, ExampleError> {
        loop {
            if self.next_segment < self.total_segments {
                let index = self.next_segment;
                self.next_segment += 1;
                let len = self.segment_lengths[index];
                if len == 0 {
                    continue;
                }

                let segment = &self.segment_storage[index][..len];
                let (_bandwidth, stereo) = self
                    .decoder
                    .decode(segment, &mut self.pcm)
                    .map_err(ExampleError::Decoder)?;

                if stereo {
                    return Err(ExampleError::UnsupportedChannels(2));
                }

                self.pcm_offset = 0;
                self.pcm_filled = PCM_BYTES_PER_FRAME;
                return Ok(true);
            }

            if !self.load_next_page()? {
                return Ok(false);
            }
        }
    }

    fn load_next_page(&mut self) -> Result<bool, ExampleError> {
        loop {
            let (segments, _) = match self.ogg.parse_next_page() {
                Ok(result) => result,
                Err(OggReaderError::Read(ReadError::UnexpectedEof)) => return Ok(false),
                Err(err) => return Err(ExampleError::Ogg(err)),
            };

            if segments.is_empty() {
                continue;
            }

            if let Some(first) = segments.get(0) {
                if first.starts_with(OPUS_TAGS_SIGNATURE) {
                    continue;
                }
            }

            let count = segments.len();
            self.total_segments = count;
            self.next_segment = 0;

            for (idx, segment) in segments.into_iter().enumerate() {
                let len = segment.len();
                self.segment_lengths[idx] = len;
                if len > 0 {
                    self.segment_storage[idx][..len].copy_from_slice(segment);
                }
            }

            for idx in count..MAX_SEGMENT_COUNT {
                self.segment_lengths[idx] = 0;
            }

            return Ok(true);
        }
    }
}

fn main() {
    if let Err(err) = run() {
        report_error(err);
        process::exit(1);
    }
}

fn run() -> Result<(), ExampleError> {
    let mut args = env::args_os();
    let _program = args.next();
    let input = args.next().ok_or(ExampleError::Usage)?;

    if args.next().is_some() {
        return Err(ExampleError::Usage);
    }

    let input_path = Path::new(&input);
    let input_file =
        File::open(input_path).map_err(|err| ExampleError::Io("open input", err.kind()))?;
    let (ogg, header) =
        OggReader::new_with(FileStream::new(input_file)).map_err(ExampleError::Ogg)?;

    if header.channels != 1 {
        return Err(ExampleError::UnsupportedChannels(header.channels));
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(ExampleError::NoOutputDevice)?;

    let desired_rate = cpal::SampleRate(header.sample_rate);
    let supported_config = device
        .default_output_config()
        .map_err(|_| ExampleError::AudioConfig)?;

    if supported_config.sample_rate() != desired_rate {
        return Err(ExampleError::UnsupportedSampleRate(header.sample_rate));
    }

    let sample_format = supported_config.sample_format();
    let config: StreamConfig = supported_config.config();

    PLAYBACK_FINISHED.store(false, Ordering::SeqCst);
    let channels = config.channels;

    if channels == 0 {
        return Err(ExampleError::UnsupportedChannels(0));
    }

    let player = Player::new(ogg);

    let stream = match (sample_format, player) {
        (SampleFormat::I16, mut player) => device
            .build_output_stream(
                &config,
                move |data: &mut [i16], _| {
                    if PLAYBACK_FINISHED.load(Ordering::SeqCst) {
                        for value in data.iter_mut() {
                            *value = 0;
                        }
                        return;
                    }

                    match player.fill_samples_i16(data, channels) {
                        Ok(done) => {
                            if done {
                                PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            eprintln!("decoder error during playback");
                            PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            for value in data.iter_mut() {
                                *value = 0;
                            }
                        }
                    }
                },
                handle_stream_error,
                None,
            )
            .map_err(|_| ExampleError::StreamBuild)?,
        (SampleFormat::F32, mut player) => device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    if PLAYBACK_FINISHED.load(Ordering::SeqCst) {
                        for value in data.iter_mut() {
                            *value = 0.0;
                        }
                        return;
                    }

                    match player.fill_samples_f32(data, channels) {
                        Ok(done) => {
                            if done {
                                PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            eprintln!("decoder error during playback");
                            PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            for value in data.iter_mut() {
                                *value = 0.0;
                            }
                        }
                    }
                },
                handle_stream_error,
                None,
            )
            .map_err(|_| ExampleError::StreamBuild)?,
        (SampleFormat::U16, mut player) => device
            .build_output_stream(
                &config,
                move |data: &mut [u16], _| {
                    if PLAYBACK_FINISHED.load(Ordering::SeqCst) {
                        for value in data.iter_mut() {
                            *value = 0u16;
                        }
                        return;
                    }

                    match player.fill_samples_u16(data, channels) {
                        Ok(done) => {
                            if done {
                                PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            eprintln!("decoder error during playback");
                            PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
                            for value in data.iter_mut() {
                                *value = 0u16;
                            }
                        }
                    }
                },
                handle_stream_error,
                None,
            )
            .map_err(|_| ExampleError::StreamBuild)?,
        _ => return Err(ExampleError::UnsupportedSampleFormat),
    };

    stream.play().map_err(|_| ExampleError::StreamStart)?;

    while !PLAYBACK_FINISHED.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(50));
    }

    stream.pause().ok();

    Ok(())
}

fn handle_stream_error(err: StreamError) {
    eprintln!("stream error: {err}");
    PLAYBACK_FINISHED.store(true, Ordering::SeqCst);
}

fn report_error(err: ExampleError) {
    match err {
        ExampleError::Usage => eprintln!("Usage: playback <in-file>"),
        ExampleError::Io(context, kind) => eprintln!("IO error ({context}): {kind:?}"),
        ExampleError::Ogg(err) => eprintln!("ogg reader error: {err}"),
        ExampleError::Decoder(err) => eprintln!("decoder error: {err}"),
        ExampleError::UnsupportedChannels(ch) => {
            eprintln!("unsupported channel count: {ch}");
        }
        ExampleError::NoOutputDevice => eprintln!("no output audio device available"),
        ExampleError::AudioConfig => eprintln!("failed to query audio configuration"),
        ExampleError::StreamBuild => eprintln!("failed to build output stream"),
        ExampleError::StreamStart => eprintln!("failed to start output stream"),
        ExampleError::UnsupportedSampleFormat => {
            eprintln!("unsupported sample format from audio device")
        }
        ExampleError::UnsupportedSampleRate(rate) => {
            eprintln!("unsupported sample rate: {rate}");
        }
    }
}

enum ExampleError {
    Usage,
    Io(&'static str, io::ErrorKind),
    Ogg(OggReaderError),
    Decoder(DecoderError),
    UnsupportedChannels(u8),
    NoOutputDevice,
    AudioConfig,
    StreamBuild,
    StreamStart,
    UnsupportedSampleFormat,
    UnsupportedSampleRate(u32),
}

impl From<OggReaderError> for ExampleError {
    fn from(value: OggReaderError) -> Self {
        Self::Ogg(value)
    }
}

impl From<DecoderError> for ExampleError {
    fn from(value: DecoderError) -> Self {
        Self::Decoder(value)
    }
}
