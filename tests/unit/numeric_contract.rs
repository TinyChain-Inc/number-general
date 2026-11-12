use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use collate::Collate;
use num::traits::Pow;
use safecast::CastFrom;

use super::{
    Boolean, Complex, ErrorKind, Float, FloatCollator, FloatInstance, Int, Number, NumberClass,
    NumberCollator, NumberInstance, UInt,
};

fn hash(value: Number) -> u64 {
    let mut state = DefaultHasher::new();
    value.hash(&mut state);
    state.finish()
}

#[test]
fn exact_numeric_equality_hashing_and_collation_laws() {
    let values: [Number; 31] = [
        false.into(),
        (-0.0f32).into(),
        0u64.into(),
        [0.0, 0.0].into(),
        true.into(),
        1i8.into(),
        1u64.into(),
        1.0f32.into(),
        1.0f64.into(),
        [1.0, 0.0].into(),
        (-1i64).into(),
        u64::MAX.into(),
        i64::MAX.into(),
        9_007_199_254_740_992u64.into(),
        9_007_199_254_740_993u64.into(),
        9_007_199_254_740_992f64.into(),
        f64::NAN.into(),
        f32::NAN.into(),
        f64::INFINITY.into(),
        f64::NEG_INFINITY.into(),
        [1.0, 1.0].into(),
        [-1.0, 1.0].into(),
        [0.0, 2.0].into(),
        0.5f64.into(),
        (-0.5f64).into(),
        (u64::MAX as f64).into(),
        (-(u64::MAX as f64)).into(),
        f64::MAX.into(),
        f64::MIN.into(),
        i64::MIN.into(),
        (i64::MIN + 1).into(),
    ];
    let collator = NumberCollator::default();
    for &left in &values {
        for &right in &values {
            assert_eq!(left == right, right == left);
            assert_eq!(
                left.partial_cmp(&right),
                right.partial_cmp(&left).map(Ordering::reverse)
            );
            assert_eq!(
                collator.cmp(&left, &right),
                collator.cmp(&right, &left).reverse()
            );
            if left == right {
                assert_eq!(left.partial_cmp(&right), Some(Ordering::Equal));
                assert_eq!(hash(left), hash(right));
                assert_eq!(collator.cmp(&left, &right), Ordering::Equal);
            }
            for third in &values {
                if collator.cmp(&left, &right).is_le() && collator.cmp(&right, third).is_le() {
                    assert!(collator.cmp(&left, third).is_le());
                }
            }
        }
    }
    assert_ne!(Number::from(-1i64), Number::from(u64::MAX));
    assert_ne!(
        Number::from(9_007_199_254_740_993u64),
        Number::from(9_007_199_254_740_992f64)
    );
    assert_ne!(Number::from(1.0000000000000002f64), Number::from(1.0f64));
    assert_eq!(
        collator.cmp(&Number::from([1.0, 1.0]), &Number::from([-1.0, 1.0])),
        Ordering::Equal
    );
    assert_ne!(Number::from([1.0, 1.0]), Number::from([-1.0, 1.0]));

    let floats: [Float; 16] = [
        f32::NEG_INFINITY.into(),
        f64::NEG_INFINITY.into(),
        (-1.0f32).into(),
        (-0.0f32).into(),
        0.0f64.into(),
        f32::from_bits(1).into(),
        1.0f32.into(),
        1.0f64.into(),
        1.0000000000000002f64.into(),
        f32::INFINITY.into(),
        f64::INFINITY.into(),
        f32::NAN.into(),
        f32::from_bits(0xffc0_0001).into(),
        f64::NAN.into(),
        f64::from_bits(0x7ff0_0000_0000_0001).into(),
        f64::from_bits(0xfff8_0000_0000_0001).into(),
    ];
    let collator = FloatCollator;
    for left in &floats {
        for right in &floats {
            let expected = match (left.is_nan(), right.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => f64::from(*left).partial_cmp(&f64::from(*right)).unwrap(),
            };
            assert_eq!(collator.cmp(left, right), expected);
            assert_eq!(
                collator.cmp(left, right),
                collator.cmp(right, left).reverse()
            );
        }
    }
}

#[test]
fn complex_magnitude_and_truthiness_preserve_finite_values() {
    for (value, expected) in [
        (
            Complex::from(num::Complex::new(3e20f32, 4e20f32)),
            Float::from(5e20f32),
        ),
        (
            Complex::from(num::Complex::new(3e-30f32, 4e-30f32)),
            Float::from(5e-30f32),
        ),
        (
            Complex::from(num::Complex::new(3e200f64, 4e200f64)),
            Float::from(5e200f64),
        ),
        (
            Complex::from(num::Complex::new(3e-200f64, 4e-200f64)),
            Float::from(5e-200f64),
        ),
        (
            Complex::from(num::Complex::new(f32::from_bits(1), 0.0)),
            Float::from(f32::from_bits(1)),
        ),
        (
            Complex::from(num::Complex::new(0.0, f64::from_bits(1))),
            Float::from(f64::from_bits(1)),
        ),
    ] {
        match (value.abs(), expected) {
            (Float::F32(actual), Float::F32(expected)) => {
                assert!(actual.is_finite());
                assert!((actual - expected).abs() <= expected * f32::EPSILON * 4.0);
            }
            (Float::F64(actual), Float::F64(expected)) => {
                assert!(actual.is_finite());
                assert!((actual - expected).abs() <= expected * f64::EPSILON * 4.0);
            }
            _ => panic!("complex magnitude must retain its component precision"),
        }
        assert!(bool::from(Boolean::cast_from(value)));
    }

    for value in [
        Complex::from(num::Complex::new(-0.0f32, 0.0)),
        Complex::from(num::Complex::new(0.0f64, -0.0)),
    ] {
        assert!(!bool::from(Boolean::cast_from(value)));
        let magnitude = f64::from(value.abs());
        assert_eq!(magnitude, 0.0);
        assert!(magnitude.is_sign_positive());
    }
    for value in [
        Complex::from(num::Complex::new(f32::NAN, 0.0)),
        Complex::from(num::Complex::new(0.0, f64::NAN)),
    ] {
        assert!(bool::from(Boolean::cast_from(value)));
        assert!(f64::from(value.abs()).is_nan());
    }
    for value in [
        Complex::from(num::Complex::new(f32::INFINITY, 0.0)),
        Complex::from(num::Complex::new(0.0, f64::NEG_INFINITY)),
    ] {
        assert!(bool::from(Boolean::cast_from(value)));
        assert!(f64::from(value.abs()).is_infinite());
    }
}

#[test]
fn integer_arithmetic_uses_promoted_widths() {
    fn signed_result(
        actual: Result<Int, super::Error>,
        expected: i128,
        bits: usize,
    ) -> Option<Int> {
        let limit = 1_i128 << (bits - 1);
        if expected < -limit {
            assert_eq!(actual.unwrap_err().kind(), ErrorKind::Underflow);
            None
        } else if expected >= limit {
            assert_eq!(actual.unwrap_err().kind(), ErrorKind::Overflow);
            None
        } else {
            let actual = actual.unwrap();
            assert_eq!(i128::from(i64::from(actual)), expected);
            assert_eq!(actual.class().size() * u8::BITS as usize, bits);
            Some(actual)
        }
    }

    fn unsigned_result(
        actual: Result<UInt, super::Error>,
        expected: Option<u128>,
        bits: usize,
    ) -> Option<UInt> {
        match expected {
            None => {
                assert_eq!(actual.unwrap_err().kind(), ErrorKind::Underflow);
                None
            }
            Some(value) if value >= 1_u128 << bits => {
                assert_eq!(actual.unwrap_err().kind(), ErrorKind::Overflow);
                None
            }
            Some(value) => {
                let actual = actual.unwrap();
                assert_eq!(u128::from(u64::from(actual)), value);
                assert_eq!(actual.class().size() * u8::BITS as usize, bits);
                Some(actual)
            }
        }
    }

    // Mixed-family operands are compared in the mathematical integer domain,
    // even when one operand alone cannot fit the selected signed destination.
    for (left, left_bits) in [
        (Int::I8(-1), 8),
        (Int::I16(-1), 16),
        (Int::I32(-1), 32),
        (Int::I64(-1), 64),
    ] {
        for (right, right_bits) in [
            (UInt::U8(u8::MAX), 16),
            (UInt::U16(u16::MAX), 16),
            (UInt::U32(u32::MAX), 32),
            (UInt::U64(u64::MAX), 64),
        ] {
            let l = i128::from(i64::from(left));
            let r = i128::from(u64::from(right));
            let bits = left_bits.max(right_bits);
            let as_int = |result: Result<Number, super::Error>| {
                result.map(|n| match n {
                    Number::Int(n) => n,
                    _ => panic!("signed result family"),
                })
            };
            signed_result(
                as_int(Number::from(left).try_add(right.into())),
                l + r,
                bits,
            );
            signed_result(
                as_int(Number::from(right).try_add(left.into())),
                r + l,
                bits,
            );
            signed_result(
                as_int(Number::from(left).try_sub(right.into())),
                l - r,
                bits,
            );
            signed_result(
                as_int(Number::from(right).try_sub(left.into())),
                r - l,
                bits,
            );
            signed_result(
                as_int(Number::from(left).try_mul(right.into())),
                l * r,
                bits,
            );
            signed_result(
                as_int(Number::from(right).try_mul(left.into())),
                r * l,
                bits,
            );
            signed_result(
                as_int(Number::from(left).try_div(right.into())),
                l / r,
                bits,
            );
            signed_result(
                as_int(Number::from(right).try_div(left.into())),
                r / l,
                bits,
            );
            signed_result(
                as_int(Number::from(left).try_rem(right.into())),
                l % r,
                bits,
            );
            signed_result(
                as_int(Number::from(right).try_rem(left.into())),
                r % l,
                bits,
            );
        }
    }

    macro_rules! signed {
        ($variant:ident, $t:ty) => {
            [<$t>::MIN, -1, 0, 1, <$t>::MAX].map(Int::$variant)
        };
    }
    let values = [
        signed!(I8, i8),
        signed!(I16, i16),
        signed!(I32, i32),
        signed!(I64, i64),
    ];
    for lefts in &values {
        for rights in &values {
            for &left in lefts {
                for &right in rights {
                    let bits = left.class().size().max(right.class().size()) * u8::BITS as usize;
                    let l = i128::from(i64::from(left));
                    let r = i128::from(i64::from(right));
                    if let Some(sum) = signed_result(left.try_add(right), l + r, bits) {
                        assert_eq!(left + right, sum);
                        assert_eq!(sum.try_sub(right).unwrap(), left);
                    }
                    if let Some(diff) = signed_result(left.try_sub(right), l - r, bits) {
                        assert_eq!(left - right, diff);
                    }
                    if let Some(product) = signed_result(left.try_mul(right), l * r, bits) {
                        let mut assigned = left;
                        assigned *= right;
                        assert_eq!(assigned, product);
                    }
                    if let Some(quotient) = l.checked_div(r) {
                        if let Some(value) = signed_result(left.try_div(right), quotient, bits) {
                            assert_eq!(left / right, value);
                        }
                        let remainder = l.checked_rem(r).unwrap();
                        let checked = signed_result(left.try_rem(right), remainder, bits).unwrap();
                        if !(l == -(1i128 << (bits - 1)) && r == -1) {
                            assert_eq!(left % right, checked);
                        }
                    } else {
                        assert_eq!(
                            left.try_div(right).unwrap_err().kind(),
                            ErrorKind::DivisionByZero
                        );
                        assert_eq!(
                            left.try_rem(right).unwrap_err().kind(),
                            ErrorKind::DivisionByZero
                        );
                    }
                    assert_eq!(i128::from(i64::from(left)), l);
                    assert_eq!(i128::from(i64::from(right)), r);
                }
            }
        }
    }

    macro_rules! unsigned {
        ($variant:ident, $t:ty) => {
            [0, 1, <$t>::MAX / 2, <$t>::MAX].map(UInt::$variant)
        };
    }
    let values = [
        unsigned!(U8, u8),
        unsigned!(U16, u16),
        unsigned!(U32, u32),
        unsigned!(U64, u64),
    ];
    for lefts in &values {
        for rights in &values {
            for &left in lefts {
                for &right in rights {
                    let bits = left.class().size().max(right.class().size()) * u8::BITS as usize;
                    let l = u128::from(u64::from(left));
                    let r = u128::from(u64::from(right));
                    if let Some(sum) = unsigned_result(left.try_add(right), Some(l + r), bits) {
                        assert_eq!(left + right, sum);
                    }
                    if let Some(diff) = unsigned_result(left.try_sub(right), l.checked_sub(r), bits)
                    {
                        let mut assigned = left;
                        assigned -= right;
                        assert_eq!(assigned, diff);
                    }
                    if let Some(product) = unsigned_result(left.try_mul(right), Some(l * r), bits) {
                        assert_eq!(left * right, product);
                    }
                    if let Some(quotient) = l.checked_div(r) {
                        assert_eq!(
                            left / right,
                            unsigned_result(left.try_div(right), Some(quotient), bits).unwrap()
                        );
                        assert_eq!(
                            left % right,
                            unsigned_result(
                                left.try_rem(right),
                                Some(l.checked_rem(r).unwrap()),
                                bits
                            )
                            .unwrap()
                        );
                    } else {
                        assert_eq!(
                            left.try_div(right).unwrap_err().kind(),
                            ErrorKind::DivisionByZero
                        );
                        assert_eq!(
                            left.try_rem(right).unwrap_err().kind(),
                            ErrorKind::DivisionByZero
                        );
                    }
                    assert_eq!(u128::from(u64::from(left)), l);
                    assert_eq!(u128::from(u64::from(right)), r);
                }
            }
        }
    }

    for value in [
        Int::I8(i8::MIN),
        Int::I16(i16::MIN),
        Int::I32(i32::MIN),
        Int::I64(i64::MIN),
    ] {
        assert_eq!(value.try_abs().unwrap_err().kind(), ErrorKind::Overflow);
        assert_eq!(
            Number::from(value).try_abs().unwrap_err().kind(),
            ErrorKind::Overflow
        );
    }
    let left = Number::from(-1_i8);
    let right = Number::from(2_u64);
    assert!(matches!(left + right, Number::Int(Int::I64(1))));
    assert!(matches!(left * right, Number::Int(Int::I64(-2))));
    assert!(matches!(
        right.try_div(left).unwrap(),
        Number::Int(Int::I64(-2))
    ));
    assert_eq!(
        Number::from(i64::MIN)
            .try_add(Number::from(u64::MAX))
            .unwrap(),
        Number::from(i64::MAX)
    );
}

#[test]
fn fallible_arithmetic_and_native_panic_variants() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let inputs = [
        (Number::from(Int::I8(7)), Number::from(Int::I64(0))),
        (Number::from(UInt::U16(7)), Number::from(UInt::U8(0))),
        (Number::from(true), Number::from(false)),
    ];
    for (left, right) in inputs {
        let before = (left, right);
        assert!(left.try_div(right).is_err());
        assert!(catch_unwind(|| left / right).is_err());
        let mut assigned = left;
        assert!(catch_unwind(AssertUnwindSafe(|| assigned /= right)).is_err());
        assert_eq!(assigned, left);
        if !matches!(left, Number::Bool(_)) {
            assert!(left.try_rem(right).is_err());
            assert!(catch_unwind(|| left % right).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| assigned %= right)).is_err());
            assert_eq!(assigned, left);
        }
        assert_eq!((left, right), before);
    }

    for left in [false, true] {
        for right in [false, true] {
            assert_eq!(
                Boolean::from(left).try_rem(right.into()).unwrap(),
                Boolean::from(left ^ right)
            );
        }
    }

    for float in [Float::F32(1.0), Float::F64(1.0)] {
        let zero = float.class().zero();
        assert_eq!(
            float.try_div(zero).unwrap_err().kind(),
            ErrorKind::DivisionByZero
        );
        assert_eq!(
            zero.try_div(zero).unwrap_err().kind(),
            ErrorKind::DivisionByZero
        );
        assert_eq!(
            float.try_rem(zero).unwrap_err().kind(),
            ErrorKind::DivisionByZero
        );
    }
    for complex in [
        Complex::C32(num::Complex::new(1.0, 0.0)),
        Complex::C64(num::Complex::new(1.0, 0.0)),
    ] {
        let zero = complex.class().zero();
        assert_eq!(
            complex.try_div(zero).unwrap_err().kind(),
            ErrorKind::DivisionByZero
        );
        assert_eq!(
            Number::from(complex)
                .try_div(zero.into())
                .unwrap_err()
                .kind(),
            ErrorKind::DivisionByZero
        );
    }

    assert_eq!(
        Number::from(1.0_f64)
            .try_div(Number::from(0_i8))
            .unwrap_err()
            .kind(),
        ErrorKind::DivisionByZero
    );
    assert_eq!(
        Number::from(1.0_f64)
            .try_rem(Number::from(0_i8))
            .unwrap_err()
            .kind(),
        ErrorKind::DivisionByZero
    );
    assert!(Int::I8(1).try_div(Int::I8(0)).is_err());
    assert!(UInt::U8(1).try_rem(UInt::U8(0)).is_err());
    assert!(Boolean::from(true).try_div(Boolean::from(false)).is_err());
}

#[test]
fn integer_powers_preserve_width_and_dynamic_promotion() {
    macro_rules! signed {
        ($variant:ident, $primitive:ty) => {{
            for base in [<$primitive>::MIN, -3, -1, 0, 1, 3, <$primitive>::MAX] {
                for exponent in [0_u32, 1, 2, 3, 63] {
                    let actual = Int::$variant(base).try_pow(Number::from(u64::from(exponent)));
                    match base.checked_pow(exponent) {
                        Some(expected) => {
                            let actual = actual.unwrap();
                            assert_eq!(actual.class(), Int::$variant(base).class());
                            assert_eq!(i64::from(actual), i64::from(expected));
                        }
                        None => assert_eq!(
                            actual.unwrap_err().kind(),
                            if base < 0 && exponent & 1 != 0 {
                                ErrorKind::Underflow
                            } else {
                                ErrorKind::Overflow
                            }
                        ),
                    }
                }
            }
            for exponent in [u64::from(u32::MAX) + 1, u64::MAX] {
                for base in [-1, 0, 1] {
                    let actual = Int::$variant(base).pow(Number::from(exponent));
                    let expected = if base == 0 {
                        0
                    } else if base == -1 && exponent & 1 != 0 {
                        -1
                    } else {
                        1
                    };
                    assert_eq!(i64::from(actual), expected);
                }
            }
            assert_eq!(
                Int::$variant(-1).pow(Number::from(-3_i64)),
                Int::$variant(-1)
            );
            assert_eq!(
                Int::$variant(-1).pow(Number::from(i64::MIN)),
                Int::$variant(1)
            );
            assert_eq!(
                Int::$variant(2)
                    .try_pow(Number::from(-1_i64))
                    .unwrap_err()
                    .kind(),
                ErrorKind::Unrepresentable
            );
            assert_eq!(Int::$variant(1).pow(Number::from(-1_i64)), Int::$variant(1));
        }};
    }
    signed!(I8, i8);
    signed!(I16, i16);
    signed!(I32, i32);
    signed!(I64, i64);

    macro_rules! unsigned {
        ($variant:ident, $primitive:ty) => {{
            for base in [0, 1, 3, <$primitive>::MAX] {
                for exponent in [0_u32, 1, 2, 3, 63] {
                    let actual = UInt::$variant(base).try_pow(Number::from(u64::from(exponent)));
                    match base.checked_pow(exponent) {
                        Some(expected) => {
                            let actual = actual.unwrap();
                            assert_eq!(actual.class(), UInt::$variant(base).class());
                            assert_eq!(u64::from(actual), u64::from(expected));
                        }
                        None => assert_eq!(actual.unwrap_err().kind(), ErrorKind::Overflow),
                    }
                }
            }
            for exponent in [u64::from(u32::MAX) + 1, u64::MAX] {
                assert_eq!(
                    UInt::$variant(0).pow(Number::from(exponent)),
                    UInt::$variant(0)
                );
                assert_eq!(
                    UInt::$variant(1).pow(Number::from(exponent)),
                    UInt::$variant(1)
                );
            }
            assert_eq!(
                UInt::$variant(1).pow(Number::from(-1_i64)),
                UInt::$variant(1)
            );
            assert_eq!(
                UInt::$variant(2)
                    .try_pow(Number::from(-1_i64))
                    .unwrap_err()
                    .kind(),
                ErrorKind::Unrepresentable
            );
        }};
    }
    unsigned!(U8, u8);
    unsigned!(U16, u16);
    unsigned!(U32, u32);
    unsigned!(U64, u64);

    for base in [Number::from(2_i8), Number::from(2_u8)] {
        let result = base.pow(Number::from(-1_i64));
        assert!(matches!(result, Number::Float(_)));
        assert_eq!(result, Number::from(0.5));
    }
    assert_eq!(
        Number::from(-1_i8).pow(Number::from(-3_i64)),
        Number::from(-1.0)
    );
    assert_eq!(
        Int::I8(3).try_pow(Number::from(2.5)).unwrap_err().kind(),
        ErrorKind::Unrepresentable
    );
}
