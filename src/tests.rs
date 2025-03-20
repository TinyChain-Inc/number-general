#![expect(
    clippy::approx_constant,
    clippy::modulo_one,
    reason = "This is a test file"
)]
use super::*;

#[test]
fn test_complex() {
    let n = Complex::from([1.23, -3.14]);
    assert_eq!(n.re(), Float::from(1.23));
    assert_eq!(n.im(), Float::from(-3.14));
}

#[test]
fn test_log() {
    let n = 1.23f32;
    assert_eq!(n.ln(), f32::cast_from(Number::from(n).ln()));
    assert_eq!(
        (-n).ln().is_nan(),
        f32::cast_from(Number::from(-n).ln()).is_nan()
    );
}

#[test]
fn test_ops() {
    let ones = [
        Number::from(true),
        Number::from(1u8),
        Number::from(1u16),
        Number::from(1u32),
        Number::from(1u64),
        Number::from(1i16),
        Number::from(1i32),
        Number::from(1i64),
        Number::from(1f32),
        Number::from(1f64),
        Number::from(_Complex::new(1f32, 0f32)),
        Number::from(_Complex::new(1f64, 0f64)),
    ];

    let f = Number::from(false);
    let t = Number::from(true);
    let two = Number::from(2);

    for one in &ones {
        let one = *one;
        let zero = one.class().zero();

        assert_eq!(one, one.class().one());

        assert_eq!(two, one * two);
        assert_eq!(one, (one * two) - one);
        assert_eq!(two, (one * two) / one);
        assert_eq!(zero, one * zero);
        assert_eq!(two.log(Float::cast_from(two)), one);
        assert_eq!(two.ln() / two.ln(), one);

        if one.is_real() {
            assert_eq!(Number::from(2 % 1), two % one);
            assert_eq!(one, one.pow(zero));
            assert_eq!(one * one, one.pow(two));
            assert_eq!(two.pow(two), (one * two).pow(two));
        }

        assert_eq!(f, one.not());
        assert_eq!(f, one.and(zero));
        assert_eq!(t, one.or(zero));
        assert_eq!(t, one.xor(zero));
    }
}

#[test]
fn test_parse() {
    assert_eq!(Number::from_str("12").unwrap(), Number::from(12));
    assert_eq!(Number::from_str("1e6").unwrap(), Number::from(1e6));
    assert_eq!(Number::from_str("1e-6").unwrap(), Number::from(1e-6));
    assert_eq!(Number::from_str("1e-06").unwrap(), Number::from(1e-6));
    assert_eq!(Number::from_str("+31.4").unwrap(), Number::from(31.4));
    assert_eq!(Number::from_str("-3.14").unwrap(), Number::from(-3.14));
    assert_eq!(
        Number::from_str("3.14+1.414i").unwrap(),
        Number::from(num::Complex::new(3.14, 1.414))
    );
    assert_eq!(
        Number::from_str("1+2i").unwrap(),
        Number::from(num::Complex::new(1., 2.))
    );
    assert_eq!(
        Number::from_str("1.0-2i").unwrap(),
        Number::from(num::Complex::new(1., -2.))
    );
}

#[cfg(feature = "serde")]
#[test]
fn test_serialize() {
    let numbers = [
        Number::from(false),
        Number::from(12u16),
        Number::from(-3),
        Number::from(3.14),
        Number::from(_Complex::<f32>::new(0., -1.414)),
    ];

    for expected in &numbers {
        let serialized = serde_json::to_string(expected).unwrap();
        let actual = serde_json::from_str(&serialized).unwrap();

        assert_eq!(expected, &actual);
    }
}
