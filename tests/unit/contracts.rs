use std::hint::black_box;
use std::panic::{catch_unwind, AssertUnwindSafe};

use num::traits::Pow;
use safecast::CastFrom;

use super::{
    Boolean, Complex, ComplexType, ErrorKind, Float, FloatInstance, FloatType, Int, IntType,
    Number, NumberClass, NumberInstance, NumberType, Trigonometry, UInt, UIntType,
};

#[test]
fn ordinary_integer_ops_follow_primitive_overflow() {
    macro_rules! dtype {
        ($t:ty) => {{
            for (a, b) in [
                (<$t>::MAX, 1 as $t),
                (<$t>::MIN, 1 as $t),
                (<$t>::MAX, <$t>::MAX),
                (<$t>::MIN, (1 as $t).wrapping_neg()),
                (1 as $t, 0 as $t),
            ] {
                macro_rules! operation {
                    ($op:tt) => {{
                        let expected = catch_unwind(|| black_box(a) $op black_box(b));
                        let actual = catch_unwind(|| {
                            Number::from(black_box(a)) $op Number::from(black_box(b))
                        });
                        assert_eq!(actual.is_err(), expected.is_err(),
                            "{} {}", stringify!($t), stringify!($op));
                        if let (Ok(actual), Ok(expected)) = (actual, expected) {
                            assert_eq!(actual, Number::from(expected));
                            assert_eq!(actual.class(), Number::from(expected).class());
                        }
                    }};
                }
                operation!(+);
                operation!(-);
                operation!(*);
                operation!(/);
                operation!(%);
            }

            let mut value = Number::from(<$t>::MAX);
            let before = value;
            let result = catch_unwind(AssertUnwindSafe(|| {
                value += Number::from(black_box(1 as $t));
            }));
            if result.is_err() {
                assert_eq!(value, before);
            }
        }};
    }
    dtype!(i8);
    dtype!(i16);
    dtype!(i32);
    dtype!(i64);
    dtype!(u8);
    dtype!(u16);
    dtype!(u32);
    dtype!(u64);
    // Mixed-family arithmetic keeps its checked mathematical result in every profile.
    assert!(catch_unwind(|| Number::from(i64::MAX) + Number::from(1u64)).is_err());
    assert_eq!(
        Number::from(i64::MIN) + Number::from(u64::MAX),
        Number::from(i64::MAX)
    );
}

#[test]
fn ordinary_ieee_math_and_checked_results_are_separate() {
    let tiny = f64::MIN_POSITIVE * f64::EPSILON;
    assert!(f64::from(Float::from(-1.).ln()).is_nan());
    assert!(f64::from(Float::from(1.) / Float::from(0.)).is_infinite());
    assert!(f64::from(Float::from(f64::MAX) * Float::from(2.)).is_infinite());
    assert_eq!(Float::from(tiny) * Float::from(0.5), Float::from(0.));
    assert_eq!(
        Float::from(tiny).try_mul(0.5.into()).unwrap_err().kind(),
        ErrorKind::Underflow
    );
    assert_eq!(
        Float::from(f64::MAX)
            .try_mul(2.0.into())
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
    let rounded = Float::from(-0.1).round();
    assert!(f64::from(rounded).is_sign_negative());
    assert_eq!(rounded.class(), FloatType::F64);
    assert!(f64::from(Float::from(f64::INFINITY).round()).is_infinite());
    assert_eq!(
        Float::from(f64::MAX).try_round().unwrap(),
        Float::from(f64::MAX)
    );
    assert_eq!(
        Number::from(8.).try_log(Number::from(2.)).unwrap(),
        Number::from(3.)
    );
    assert_eq!(
        Float::from(8f32)
            .try_log(Float::from(2f64))
            .unwrap()
            .class(),
        FloatType::F64
    );
    assert_eq!(Boolean::from(true).try_acos().unwrap(), Float::from(0f32));
    assert_eq!(
        Boolean::from(true).try_sin().unwrap(),
        Float::from(1f32).try_sin().unwrap()
    );
}

#[test]
fn exact_powers_preserve_bases_and_integral_exponents() {
    let bool_negative = Number::from(true).try_pow(Number::from(-1i8)).unwrap();
    assert_eq!(bool_negative, Number::from(1f32));
    assert_eq!(bool_negative.class(), FloatType::F32.into());

    for (base, exp) in [
        (Int::from(16_777_217i32), Number::from(1f32)),
        (Int::from(9_007_199_254_740_993i64), Number::from(1f64)),
    ] {
        let result = base.try_pow(exp).unwrap();
        assert_eq!(result, base);
        assert_eq!(result.class(), base.class());
    }
    assert_eq!(
        UInt::from(u64::MAX).try_pow(1f64.into()).unwrap(),
        UInt::from(u64::MAX)
    );
    assert_eq!(
        Int::from(2i64).try_pow(53.5f64.into()).unwrap_err().kind(),
        ErrorKind::Unrepresentable
    );
    assert_eq!(
        Int::from(1i8).try_pow(f64::MAX.into()).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    for base in [Float::from(-1f32), Float::from(-1f64)] {
        assert_eq!(base.try_pow(u64::MAX.into()).unwrap(), base);
        assert_eq!(base.pow(u64::MAX.into()), base);
        assert_eq!(base.try_pow(i64::MIN.into()).unwrap(), Float::from(1f32));
    }
    let complex = Complex::from([-1f64, 0.]);
    assert_eq!(complex.try_pow(u64::MAX.into()).unwrap(), complex);
    assert_eq!(complex.pow(u64::MAX.into()), complex);
    assert_eq!(
        Boolean::from(false)
            .try_pow((-1).into())
            .unwrap_err()
            .kind(),
        ErrorKind::DivisionByZero
    );
    assert_eq!(
        Boolean::from(true).try_pow((-1).into()).unwrap(),
        Boolean::from(true)
    );
    assert_eq!(
        Number::from(2i8).try_pow((-1i64).into()).unwrap(),
        Number::from(0.5f32)
    );
    assert!(f64::from(Float::cast_from(Number::from(0i8).pow((-1i64).into()))).is_infinite());
}

#[test]
fn promotion_targets_and_narrowing_are_explicit() {
    assert_eq!(
        Number::from(127i8).try_add(true.into()).unwrap().class(),
        NumberType::Int(IntType::I16)
    );
    assert_eq!(
        Number::from(127i8).try_add(true.into()).unwrap(),
        Number::from(128i16)
    );
    for source in [
        Number::from(true),
        Number::from(1i8),
        Number::from(1i64),
        Number::from(1u64),
        Number::from(1f64),
    ] {
        for target in [
            NumberType::Int(IntType::Int),
            NumberType::Float(FloatType::Float),
            NumberType::Complex(ComplexType::Complex),
        ] {
            let a = target.try_cast(source).unwrap();
            let b = source.try_into_type(target).unwrap();
            assert_eq!(a.class(), b.class());
            assert_eq!(target.cast(source).class(), a.class());
            assert_eq!(source.into_type(target).class(), a.class());
        }
    }
    let converted = Int::try_from(UInt::from(u16::MAX)).unwrap();
    assert_eq!(converted, Int::I64(i64::from(u16::MAX)));
    assert_eq!(converted.class(), IntType::I64);
    assert_eq!(
        IntType::I16
            .try_cast(Number::from(u16::MAX))
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
    assert_eq!(
        IntType::Int
            .try_cast(Number::from(u16::MAX))
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
    for value in [i64::MAX as u64 + 1, u64::MAX] {
        assert_eq!(
            Int::try_from(UInt::from(value)).unwrap_err().kind(),
            ErrorKind::Overflow
        );
    }
    assert_eq!(
        Int::try_from(UInt::from(i64::MAX as u64)).unwrap(),
        Int::I64(i64::MAX)
    );
    assert_eq!(Int::cast_from(UInt::from(u16::MAX)), Int::from(-1i16));
    assert_eq!(Int::try_from(UInt::from(1u8)).unwrap(), Int::from(1i64));
    assert_eq!(usize::try_from(UInt::from(1u8)).unwrap(), 1usize);
    let result = usize::try_from(UInt::from(u64::MAX));
    if usize::BITS < u64::BITS {
        assert_eq!(result.unwrap_err().kind(), ErrorKind::Overflow);
    } else {
        assert_eq!(result.unwrap(), usize::MAX);
    }
    assert_eq!(
        Int::try_sum([Int::from(1i8)].into_iter()).unwrap().class(),
        IntType::I8
    );
    assert_eq!(
        Int::try_product([Int::from(1i8)].into_iter())
            .unwrap()
            .class(),
        IntType::I8
    );
    assert_eq!(
        Int::try_sum([Int::from(127i8), Int::from(1i8)].into_iter())
            .unwrap_err()
            .kind(),
        ErrorKind::Overflow
    );
}

#[test]
fn complex_axis_results_preserve_small_values_and_branch_sides() {
    // Cubic corrections are far below one ULP here; the rounded result is tiny.
    for tiny in [1e-300f64, -1e-300f64] {
        assert_eq!(
            Complex::from([tiny, 0.]).try_asinh().unwrap(),
            Complex::from([tiny, 0.])
        );
        assert_eq!(
            Complex::from([0., tiny]).try_asinh().unwrap(),
            Complex::from([0., tiny])
        );
        assert_eq!(
            Complex::from([0., tiny]).try_asin().unwrap(),
            Complex::from([0., tiny])
        );
    }
    assert_eq!(
        Complex::from([f64::MAX, 0.])
            .try_div(Complex::from([f64::MAX, 0.]))
            .unwrap(),
        Complex::from([1f64, 0.])
    );
    assert_eq!(
        Complex::from([1f64, 0.])
            .try_div(Complex::from([1e-300f64, 0.]))
            .unwrap(),
        Complex::from([1f64 / 1e-300, 0.])
    );
    for im in [0.0f64, -0.0] {
        let logarithm = Complex::from([-1f64, im]).try_ln().unwrap();
        assert_eq!(
            f64::from(logarithm.im()).is_sign_negative(),
            im.is_sign_negative()
        );
        assert_eq!(f64::from(logarithm.im()).abs(), std::f64::consts::PI);
        let result = Complex::from([1f64, im]).try_acos().unwrap();
        assert_eq!(
            f64::from(result.im()).is_sign_negative(),
            !im.is_sign_negative()
        );
    }
    let base = Complex::from([2f64, 0.]);
    assert_eq!(
        Complex::from([8f64, 0.]).try_log(base).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let actual = Complex::from([8f64, 0.]).log(base);
    assert!((f64::from(actual.re()) - 3.).abs() < 1e-14);
    assert_eq!(
        Complex::from([2f64, 0.]).pow(Number::from([0f64, 0.])),
        Complex::from([1f64, 0.])
    );
}

#[test]
fn wrapping_arithmetic_preserves_numeric_promotion() {
    // Independent promotion metadata and widened mathematical expectations.
    let mut cases = Vec::new();
    macro_rules! cases {
        ($t:ty, $signed:expr) => {{
            for value in [<$t>::MIN, <$t>::MAX, 0, 1, (1 as $t).wrapping_neg()] {
                cases.push((Number::from(value), <$t>::BITS, $signed, value as i128));
            }
        }};
    }
    cases!(i8, true);
    cases!(i16, true);
    cases!(i32, true);
    cases!(i64, true);
    cases!(u8, false);
    cases!(u16, false);
    cases!(u32, false);
    cases!(u64, false);

    for &(left, lbits, lsigned, l) in &cases {
        for &(right, rbits, rsigned, r) in &cases {
            let signed = lsigned || rsigned;
            let signed_width =
                |bits: u32, is_signed: bool| if !is_signed && bits == 8 { 16 } else { bits };
            let bits = if signed {
                signed_width(lbits, lsigned).max(signed_width(rbits, rsigned))
            } else {
                lbits.max(rbits)
            };
            let dtype = if signed {
                NumberType::Int(match bits {
                    8 => IntType::I8,
                    16 => IntType::I16,
                    32 => IntType::I32,
                    64 => IntType::I64,
                    _ => unreachable!(),
                })
            } else {
                NumberType::UInt(match bits {
                    8 => UIntType::U8,
                    16 => UIntType::U16,
                    32 => UIntType::U32,
                    64 => UIntType::U64,
                    _ => unreachable!(),
                })
            };
            for (name, operation) in [
                ("add", Number::wrapping_add as fn(Number, Number) -> Number),
                ("sub", Number::wrapping_sub),
                ("mul", Number::wrapping_mul),
                ("div", Number::wrapping_div),
                ("rem", Number::wrapping_rem),
            ] {
                if r == 0 && matches!(name, "div" | "rem") {
                    assert!(catch_unwind(|| operation(left, right)).is_err());
                    continue;
                }
                let mathematical = if signed {
                    (match name {
                        "add" => l + r,
                        "sub" => l - r,
                        "mul" => l * r,
                        "div" => l / r,
                        "rem" => l % r,
                        _ => unreachable!(),
                    }) as u128
                } else {
                    let (l, r) = (l as u128, r as u128);
                    match name {
                        "add" => l + r,
                        "sub" => l.wrapping_sub(r),
                        "mul" => l * r,
                        "div" => l / r,
                        "rem" => l % r,
                        _ => unreachable!(),
                    }
                };
                let low = mathematical & ((1u128 << bits) - 1);
                let expected = if signed && low >= 1u128 << (bits - 1) {
                    Number::from((low as i128 - (1i128 << bits)) as i64)
                } else {
                    Number::from(low as u64)
                };
                let actual = operation(left, right);
                assert_eq!(actual, expected, "{left} {name} {right}");
                assert_eq!(actual.class(), dtype);

                macro_rules! typed {
                    ($ty:ident) => {
                        if let (Number::$ty(left), Number::$ty(right)) = (left, right) {
                            let result = match name {
                                "add" => left.wrapping_add(right),
                                "sub" => left.wrapping_sub(right),
                                "mul" => left.wrapping_mul(right),
                                "div" => left.wrapping_div(right),
                                "rem" => left.wrapping_rem(right),
                                _ => unreachable!(),
                            };
                            assert_eq!(Number::from(result), actual);
                            assert_eq!(Number::from(result).class(), dtype);
                        }
                    };
                }
                typed!(Int);
                typed!(UInt);
            }
        }
    }

    macro_rules! abs {
        ($t:ty, $ty:ident) => {{
            let value = <$t>::MIN;
            let expected = if (value as i128) < 0 {
                value.wrapping_neg()
            } else {
                value
            };
            assert_eq!(Number::from(value).wrapping_abs(), Number::from(expected));
            assert_eq!($ty::from(value).wrapping_abs(), $ty::from(expected));
            let zero = $ty::from(0 as $t);
            let one = $ty::from(1 as $t);
            assert!(catch_unwind(|| one.wrapping_div(zero)).is_err());
            assert!(catch_unwind(|| one.wrapping_rem(zero)).is_err());
        }};
    }
    abs!(i8, Int);
    abs!(i16, Int);
    abs!(i32, Int);
    abs!(i64, Int);
    abs!(u8, UInt);
    abs!(u16, UInt);
    abs!(u32, UInt);
    abs!(u64, UInt);

    assert_eq!(
        Number::from(127i8).wrapping_add(true.into()),
        Number::from(128i16)
    );
    assert_eq!(
        Number::from(127i8).wrapping_add(true.into()).class(),
        IntType::I16.into()
    );
    assert_eq!(
        Number::from(i64::MIN).wrapping_add(u64::MAX.into()),
        Number::from(i64::MAX)
    );
    assert_eq!(
        Number::from(true).wrapping_sub(true.into()),
        Number::from(false)
    );
    assert!(catch_unwind(|| Number::from(true).wrapping_div(false.into())).is_err());
    assert_eq!(
        Number::from(255u8).wrapping_add(0.5f64.into()),
        Number::from(255.5f64)
    );
    assert!(Number::from(f64::MAX)
        .wrapping_mul(2f64.into())
        .is_infinite());
    assert!(Number::from(0f64).wrapping_div(0f64.into()).is_nan());
    let complex = Number::from([2f32, 3.]);
    assert_eq!(complex.wrapping_mul(2i8.into()), Number::from([4f32, 6.]));
    assert_eq!(Number::from([3f64, 4.]).wrapping_abs(), Number::from(5f64));
    assert_eq!(Number::from(f64::NAN), Number::from(f64::NAN));
}

#[test]
fn wrapping_powers_and_aggregates_preserve_numeric_contracts() {
    macro_rules! powers {
        ($t:ty, $ty:ident) => {{
            for base in [<$t>::MIN, <$t>::MAX, 0, 1, 2] {
                for exponent in [0u32, 1, 2, 3, 4, 64] {
                    let expected = base.wrapping_pow(exponent);
                    let typed = $ty::from(base).wrapping_pow(u64::from(exponent));
                    let dynamic =
                        Number::from(base).wrapping_pow(Number::from(u64::from(exponent)));
                    assert_eq!(typed, $ty::from(expected));
                    assert_eq!(dynamic, Number::from(expected));
                    assert_eq!(typed.class(), $ty::from(base).class());
                    assert_eq!(dynamic.class(), Number::from(base).class());
                }
            }
            let values = [$ty::from(<$t>::MAX), $ty::from(1 as $t)];
            assert_eq!(
                $ty::wrapping_sum(values.into_iter()),
                $ty::from(<$t>::MAX.wrapping_add(1))
            );
            assert_eq!(
                $ty::wrapping_product(values.into_iter()),
                $ty::from(<$t>::MAX)
            );
            assert_eq!(
                $ty::wrapping_sum(std::iter::empty()),
                std::iter::empty::<$ty>().sum()
            );
            assert_eq!(
                $ty::wrapping_product(std::iter::empty()),
                std::iter::empty::<$ty>().product()
            );
        }};
    }
    powers!(i8, Int);
    powers!(i16, Int);
    powers!(i32, Int);
    powers!(i64, Int);
    powers!(u8, UInt);
    powers!(u16, UInt);
    powers!(u32, UInt);
    powers!(u64, UInt);
    assert_eq!(Int::from(-1i8).wrapping_pow(u64::MAX), Int::from(-1i8));
    assert_eq!(UInt::from(1u8).wrapping_pow(u64::MAX), UInt::from(1u8));
    assert_eq!(
        Number::from(-1i8).wrapping_pow(u64::MAX.into()),
        Number::from(-1i8)
    );
    assert_eq!(
        Number::from(2i8).wrapping_pow((-1i8).into()),
        Number::from(0.5f32)
    );
    assert_eq!(
        Number::from(4i8).wrapping_pow(0.5f32.into()),
        Number::from(2f32)
    );
    assert!(Number::from(0i8).wrapping_pow((-1i8).into()).is_infinite());
    assert_eq!(
        Number::from([-1f64, 0.]).wrapping_pow(u64::MAX.into()),
        Number::from([-1f64, 0.])
    );
    assert_eq!(
        Number::wrapping_sum([Number::from(127i8), Number::from(1u8)].into_iter()),
        Number::from(128i16)
    );
    assert_eq!(
        Number::wrapping_sum([Number::from(127i8)].into_iter()).class(),
        IntType::I8.into()
    );
    assert_eq!(
        Number::wrapping_sum(std::iter::empty()).class(),
        std::iter::empty::<Number>().sum::<Number>().class()
    );
    assert_eq!(
        Number::wrapping_product(std::iter::empty()),
        std::iter::empty::<Number>().product::<Number>()
    );
    assert_eq!(
        Number::wrapping_product([Number::from(0i8), Number::from(2f64)].into_iter()).class(),
        FloatType::F64.into()
    );
    assert!(
        Number::wrapping_product([Number::from(0i8), Number::from(f64::INFINITY)].into_iter())
            .is_nan()
    );

    struct Release<'a>(&'a std::cell::Cell<bool>);
    impl Drop for Release<'_> {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let released = std::cell::Cell::new(false);
    let visited = std::cell::Cell::new(0);
    let release = Release(&released);
    let visits = &visited;
    let values = [Number::from(0i8), Number::from(2f64)]
        .into_iter()
        .inspect(move |_| {
            let _ = &release;
            visits.set(visits.get() + 1);
        });
    Number::wrapping_product(values);
    assert_eq!(visited.get(), 2);
    assert!(released.get());
    released.set(false);
    let release = Release(&released);
    let values = [Number::from(1i8)].into_iter().inspect(move |_| {
        let _ = &release;
        panic!("iterator failure");
    });
    assert!(catch_unwind(AssertUnwindSafe(|| Number::wrapping_sum(values))).is_err());
    assert!(released.get());
}
