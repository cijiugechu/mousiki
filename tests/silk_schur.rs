use mousiki::silk::schur::silk_schur;

#[test]
fn schur_matches_reference_case_one() {
    let mut rc = [0i16; 4];
    let c = [134_217_728, 33_554_432, -16_777_216, 8_388_608, -4_194_304];
    let res = silk_schur(&mut rc, &c, 4);
    assert_eq!(460_180_984, res);
    assert_eq!([-8192, 6553, -5461, 4680], rc);
}

#[test]
fn schur_matches_reference_case_two() {
    let mut rc = [0i16; 6];
    let c = [
        1_056_964_608,
        -268_435_456,
        134_217_728,
        -67_108_864,
        33_554_432,
        -16_777_216,
        8_388_608,
    ];
    let res = silk_schur(&mut rc, &c, 6);
    assert_eq!(984_051_516, res);
    assert_eq!([8322, -2188, 578, -152, 40, -10], rc);
}

#[test]
fn schur_limits_when_unstable() {
    let mut rc = [0i16; 3];
    let c = [268_435_456, 134_217_728, 67_108_864, 33_554_432];
    let res = silk_schur(&mut rc, &c, 3);
    assert_eq!(402_653_184, res);
    assert_eq!([-16384, 0, 0], rc);
}
