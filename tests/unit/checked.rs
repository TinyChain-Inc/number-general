use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{
    Boolean, Complex, ComplexType, Error, ErrorKind, Float, FloatType, Int, IntType, Number,
    NumberClass, NumberInstance, NumberType, Trigonometry, UInt, UIntType,
};

#[path = "../support/portable.rs"]
mod portable;

fn kind<T: std::fmt::Debug>(result: Result<T, Error>, expected: ErrorKind) {
    assert_eq!(result.unwrap_err().kind(), expected);
}

#[test]
fn checked_conversions_preserve_values_and_classify_loss() {
    for dtype in [IntType::I8, IntType::I16, IntType::I32, IntType::I64] {
        let bits = dtype.size() * 8;
        let min = -(1_i128 << (bits - 1));
        let max = (1_i128 << (bits - 1)) - 1;
        for value in [min, 0, max] {
            assert_eq!(
                i64::from(dtype.try_cast(Number::from(value as i64)).unwrap()),
                value as i64
            );
        }
        kind(dtype.try_cast(Number::from(u64::MAX)), ErrorKind::Overflow);
        if bits < 64 {
            kind(
                dtype.try_cast(Number::from((min - 1) as i64)),
                ErrorKind::Underflow,
            );
        }
        kind(
            dtype.try_cast(Number::from(0.5)),
            ErrorKind::Unrepresentable,
        );
        kind(
            dtype.try_cast(Number::from(f64::INFINITY)),
            ErrorKind::Unrepresentable,
        );
    }
    for dtype in [UIntType::U8, UIntType::U16, UIntType::U32, UIntType::U64] {
        let max = ((1_u128 << (dtype.size() * 8)) - 1) as u64;
        assert_eq!(u64::from(dtype.try_cast(max.into()).unwrap()), max);
        kind(dtype.try_cast((-1).into()), ErrorKind::Underflow);
        kind(dtype.try_cast(0.5.into()), ErrorKind::Unrepresentable);
    }
    assert_eq!(
        FloatType::F32.try_cast(1.1f64.into()).unwrap(),
        Float::from(1.1f32)
    );
    kind(
        FloatType::F32.try_cast(f64::MAX.into()),
        ErrorKind::Overflow,
    );
    kind(
        FloatType::F32.try_cast(f64::MIN_POSITIVE.into()),
        ErrorKind::Underflow,
    );
    let tiny = f32::MIN_POSITIVE * f32::EPSILON;
    assert_eq!(
        FloatType::F32.try_cast(f64::from(tiny).into()).unwrap(),
        Float::from(tiny)
    );
    assert!(f64::from(FloatType::F32.try_cast((-0.0f64).into()).unwrap()).is_sign_negative());
    assert!(f64::from(FloatType::F32.try_cast(f64::NAN.into()).unwrap()).is_nan());
    kind(
        NumberType::Bool.try_cast(2.into()),
        ErrorKind::Unrepresentable,
    );
    for value in [0, 1] {
        assert_eq!(
            NumberType::Bool.try_cast(value.into()).unwrap(),
            Number::from(value != 0)
        );
    }
    kind(
        IntType::I32.try_cast(Number::from([1., 1.])),
        ErrorKind::Unrepresentable,
    );
    kind(
        ComplexType::C32.try_cast(Number::from([1., f64::MIN_POSITIVE])),
        ErrorKind::Underflow,
    );
    assert_eq!(
        IntType::Int.try_cast(1i64.into()).unwrap().class(),
        IntType::I64
    );
    assert_eq!(
        Int::from(1i64).try_into_type(IntType::Int).unwrap().class(),
        IntType::I64
    );
    // The explicitly lossy conversions remain available.
    assert_eq!(IntType::I8.cast(257.into()), Int::from(1i8));
    assert_eq!(Error::new("parse context").kind(), ErrorKind::InvalidInput);
    assert_eq!(
        Error::with_kind(ErrorKind::Domain, "context").to_string(),
        "context"
    );
}

#[test]
fn checked_real_math_distinguishes_underflow_and_valid_zeros() {
    for tiny in [
        Float::from(f32::MIN_POSITIVE * f32::EPSILON),
        Float::from(f64::MIN_POSITIVE * f64::EPSILON),
    ] {
        let one = tiny.class().one();
        assert_eq!(tiny.try_mul(one).unwrap(), tiny);
        kind(tiny.try_mul(Float::from(0.5f32)), ErrorKind::Underflow);
        assert_eq!(tiny.try_sub(tiny).unwrap(), Float::from(0.));
        assert_eq!(tiny.try_mul(Float::from(0.)).unwrap(), Float::from(0.));
        assert_eq!(tiny.try_rem(tiny).unwrap(), Float::from(0.));
    }
    kind(
        Float::from(f64::MAX).try_add(f64::MAX.into()),
        ErrorKind::Overflow,
    );
    kind(
        Float::from(f32::MAX).try_mul(2f32.into()),
        ErrorKind::Overflow,
    );
    kind(Float::from(-1000.).try_exp(), ErrorKind::Underflow);
    kind(Float::from(1000.).try_exp(), ErrorKind::Overflow);
    kind(Float::from(-1.).try_ln(), ErrorKind::Domain);
    kind(Float::from(1.).try_log(Float::from(1.)), ErrorKind::Domain);
    kind(Float::from(-2.).try_pow(0.5.into()), ErrorKind::Domain);
    kind(
        Float::from(0.).try_pow((-1).into()),
        ErrorKind::DivisionByZero,
    );
    kind(
        Float::from(2.).try_pow(Number::from([1., 1.])),
        ErrorKind::Unsupported,
    );
    assert_eq!(Float::from(1.).try_ln().unwrap(), Float::from(0.));
    assert_eq!(
        Float::from(1.).try_log(Float::from(2.)).unwrap(),
        Float::from(0.)
    );
    assert_eq!(Float::from(0.).try_pow(2.into()).unwrap(), Float::from(0.));
    assert_eq!(Float::from(0.1).try_round().unwrap(), Float::from(0.));
    assert_eq!(Int::from(1).try_div(2.into()).unwrap(), Int::from(0));
    assert_eq!(
        Float::from(f64::MAX).try_round().unwrap(),
        Float::from(f64::MAX)
    );
    kind(
        IntType::I64.try_cast(Float::from(f64::MAX).try_round().unwrap().into()),
        ErrorKind::Overflow,
    );
    assert!(f64::from(Float::from(f64::INFINITY).try_mul(0.0.into()).unwrap()).is_nan());
    assert!(f64::from(Float::from(f64::NAN).try_ln().unwrap()).is_nan());
    assert!(f64::from(Float::from(f64::NEG_INFINITY).try_exp().unwrap()) == 0.);
    let zero = Float::from(-0.0).try_mul(1.0.into()).unwrap();
    assert!(f64::from(zero).is_sign_negative());
    kind(Float::from(2.).try_asin(), ErrorKind::Domain);
    kind(Float::from(2.).try_acos(), ErrorKind::Domain);
    kind(Float::from(0.).try_acosh(), ErrorKind::Domain);
    kind(Float::from(1.).try_atanh(), ErrorKind::Domain);
    for result in [
        Float::from(0.).try_sin(),
        Float::from(0.).try_sinh(),
        Float::from(0.).try_asin(),
        Float::from(0.).try_asinh(),
        Float::from(0.).try_atan(),
        Float::from(0.).try_tan(),
        Float::from(0.).try_tanh(),
        Float::from(0.).try_atanh(),
        Float::from(1.).try_acos(),
        Float::from(1.).try_acosh(),
    ] {
        assert_eq!(result.unwrap(), Float::from(0.));
    }
    assert_eq!(Float::from(0.).try_cos().unwrap(), Float::from(1.));
    assert_eq!(Float::from(0.).try_cosh().unwrap(), Float::from(1.));
}

#[test]
fn checked_complex_math_certifies_cancellation_and_axis_zeros() {
    let c = |re, im| Complex::from(num::Complex::new(re, im));
    assert_eq!(c(1f64, 1.).try_mul(c(1., -1.)).unwrap(), c(2., 0.));
    assert_eq!(
        Complex::from([1f32, 1.])
            .try_sub(Complex::from([1f32, 1.]))
            .unwrap(),
        c(0., 0.)
    );
    assert_eq!(
        Complex::from([5f32, 3.])
            .try_rem(Complex::from([2f32, 1.]))
            .unwrap(),
        c(1., 1.)
    );
    let tiny32 = f32::MIN_POSITIVE * f32::EPSILON;
    kind(
        Complex::from([1f32, tiny32]).try_mul(Complex::from([0.25f32, 0.])),
        ErrorKind::Underflow,
    );
    let tiny = f64::MIN_POSITIVE * f64::EPSILON;
    kind(c(1f64, tiny).try_mul(c(0.25, 0.)), ErrorKind::Underflow);
    kind(c(tiny, tiny).try_mul(c(0.25, 0.)), ErrorKind::Underflow);
    kind(c(1f64, 1e-200).try_ln(), ErrorKind::Unsupported);
    kind(c(0f64, 0.).try_ln(), ErrorKind::Domain);
    kind(c(1f64, 1.).try_div(c(0., 0.)), ErrorKind::DivisionByZero);
    kind(
        c(1f64, 1.).try_pow(Number::from([1., 0.])),
        ErrorKind::Unsupported,
    );
    assert_eq!(c(1f64, 1.).try_pow(0.into()).unwrap(), c(1., 0.));
    assert_eq!(c(1f64, 0.).try_ln().unwrap(), c(0., 0.));
    assert_eq!(c(0f64, 0.).try_exp().unwrap(), c(1., 0.));
    assert_eq!(c(0f64, 0.).try_sin().unwrap(), c(0., 0.));
    assert_eq!(c(0f64, 0.).try_cos().unwrap(), c(1., 0.));
    assert_eq!(c(0f64, 0.).try_sinh().unwrap(), c(0., 0.));
    assert_eq!(c(0f64, 0.).try_cosh().unwrap(), c(1., 0.));
    assert_eq!(c(tiny, 0.).try_abs().unwrap(), Float::from(tiny));
    assert_eq!(c(0.1f64, -0.1).try_round().unwrap(), c(0., -0.));
    kind(c(-1000f64, 0.).try_exp(), ErrorKind::Underflow);
    assert!(f64::from(c(f64::INFINITY, 0.).try_abs().unwrap()).is_infinite());
    let original = c(1f64, tiny);
    assert!(original.try_mul(c(0.25, 0.)).is_err());
    assert_eq!(original, c(1., tiny));
}

#[test]
fn checked_aggregates_preserve_order_and_release_iterators() {
    assert_eq!(
        Number::try_sum(std::iter::empty()).unwrap(),
        Number::from(false)
    );
    assert_eq!(
        Number::try_product(std::iter::empty()).unwrap(),
        Number::from(true)
    );
    assert_eq!(Int::try_sum(std::iter::empty()).unwrap(), Int::from(0i16));
    assert_eq!(
        UInt::try_product(std::iter::empty()).unwrap(),
        UInt::from(1u8)
    );
    kind(
        Number::try_sum(
            [
                Number::from(u64::MAX),
                Number::from(1u64),
                Number::from(-1i64),
            ]
            .into_iter(),
        ),
        ErrorKind::Overflow,
    );
    let seen = Cell::new(0);
    let dropped = Cell::new(false);

    struct Inputs<'a> {
        seen: &'a Cell<usize>,
        dropped: &'a Cell<bool>,
    }

    impl Iterator for Inputs<'_> {
        type Item = Number;

        fn next(&mut self) -> Option<Number> {
            let values = [
                Number::from(u64::MAX),
                Number::from(2u64),
                Number::from(0u64),
            ];
            let next = values.get(self.seen.get()).copied();

            if next.is_some() {
                self.seen.set(self.seen.get() + 1);
            }

            next
        }
    }

    impl Drop for Inputs<'_> {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    kind(
        Number::try_product(Inputs {
            seen: &seen,
            dropped: &dropped,
        }),
        ErrorKind::Overflow,
    );
    assert_eq!(seen.get(), 2);
    assert!(dropped.get());
    let result = Number::try_product([Number::from(0i8), Number::from(2f64)].into_iter()).unwrap();
    assert_eq!(result.class(), NumberType::Float(FloatType::F64));
    assert_eq!(result, Number::from(0.));
    assert!(
        f64::from(Float::try_product([0.0.into(), f64::INFINITY.into()].into_iter()).unwrap())
            .is_nan()
    );
    seen.set(0);
    assert_eq!(
        Boolean::try_product([false, true].into_iter().map(|x| {
            seen.set(seen.get() + 1);
            Boolean::from(x)
        }))
        .unwrap(),
        Boolean::from(false)
    );
    assert_eq!(seen.get(), 1);
}

#[test]
fn native_panics_leave_assignments_unchanged() {
    let mut value = Int::from(i8::MAX);
    let primitive = catch_unwind(|| std::hint::black_box(i8::MAX) + std::hint::black_box(1i8));
    let outcome = catch_unwind(AssertUnwindSafe(|| value += Int::from(1i8)));
    assert_eq!(outcome.is_err(), primitive.is_err());
    assert_eq!(
        value,
        primitive.map(Int::from).unwrap_or(Int::from(i8::MAX))
    );
    let mut value = UInt::from(0u8);
    let primitive = catch_unwind(|| std::hint::black_box(0u8) - std::hint::black_box(1u8));
    let outcome = catch_unwind(AssertUnwindSafe(|| value -= UInt::from(1u8)));
    assert_eq!(outcome.is_err(), primitive.is_err());
    assert_eq!(value, primitive.map(UInt::from).unwrap_or(UInt::from(0u8)));
    let mut value = Float::from(f32::MAX);
    kind(value.try_mul(Float::from(2f32)), ErrorKind::Overflow);
    assert_eq!(value, Float::from(f32::MAX));
    value *= Float::from(2f32);
    assert!(f64::from(value).is_infinite());
    let mut value = Int::from(i64::MIN);
    assert!(catch_unwind(AssertUnwindSafe(|| value /= Int::from(-1i64))).is_err());
    assert_eq!(value, Int::from(i64::MIN));
    assert_eq!(value.try_rem(Int::from(-1i64)).unwrap(), Int::from(0));
    assert!(f64::from(Float::from(-1.).ln()).is_nan());
    assert!(f64::from(Float::from(2.).asin()).is_nan());
}

#[test]
fn portable_real_kernel_vectors() {
    portable::portable_real_kernel_vectors();
}

#[test]
fn portable_preparation_preserves_values_and_types() {
    let input = Number::from(16_777_217i32);
    let base = Number::from(2f64);
    let result = input.try_log(base).unwrap();
    assert_eq!(result.class(), FloatType::F64.into());
    assert_eq!(
        result,
        Float::from(16_777_217f64)
            .try_log(Float::from(2f64))
            .unwrap()
            .into()
    );
    assert_ne!(result, Number::from(24f64));
    let Number::Float(Float::F64(logarithm)) = result else {
        panic!("dtype")
    };
    assert_eq!(logarithm.to_bits(), 0x4038000001715476);
    let Float::F32(power) = Float::from(1.0001f32)
        .try_pow(Number::from(1.0002f32))
        .unwrap()
    else {
        panic!("dtype")
    };
    assert_eq!(power.to_bits(), 0x3f800347);
    let Float::F64(power) = Float::from(1.0001f64)
        .try_pow(Number::from(1.0002f64))
        .unwrap()
    else {
        panic!("dtype")
    };
    assert_eq!(power.to_bits(), 0x3ff00068e0ea21c2);

    assert_eq!(
        Int::from(16_777_217i32).try_log(Float::from(2f64)).unwrap(),
        Float::from(16_777_217f64)
            .try_log(Float::from(2f64))
            .unwrap()
    );
    assert_eq!(input.log(base), Number::from(16_777_217f64).log(base));
    assert_eq!(
        Int::from(16_777_217i32).log(Float::from(2f64)),
        Float::from(16_777_217f64).log(Float::from(2f64))
    );
    assert_eq!(
        UInt::from(16_777_217u32)
            .try_log(Float::from(2f64))
            .unwrap(),
        Float::from(16_777_217f64)
            .try_log(Float::from(2f64))
            .unwrap()
    );
    assert_eq!(
        Boolean::from(true)
            .try_log(Float::from(2f64))
            .unwrap()
            .class(),
        FloatType::F64
    );

    for integer in [
        Number::from(16_777_217i32),
        Number::from(9_007_199_254_740_993i64),
        Number::from(u64::MAX),
    ] {
        for exponent in [
            Number::from(1u8),
            Number::from(1i64),
            Number::from(1f32),
            Number::from(1f64),
        ] {
            for value in [
                integer.pow(exponent),
                integer.try_pow(exponent).unwrap(),
                integer.wrapping_pow(exponent),
            ] {
                assert_eq!(value, integer);
                assert_eq!(value.class(), integer.class());
            }
        }
    }
    for exponent in [Number::from(2i8), Number::from(2f64)] {
        let integer = Number::from(127i8);
        kind(integer.try_pow(exponent), ErrorKind::Overflow);
        assert_eq!(integer.wrapping_pow(exponent), Number::from(1i8));
    }
    kind(
        Number::from(1i8).try_pow(f64::MAX.into()),
        ErrorKind::Unsupported,
    );
    let real = Number::from(1f64);
    assert_eq!(
        real.wrapping_pow(f64::MAX.into()),
        real.pow(f64::MAX.into())
    );
    let complex = Number::from([1f64, 0.]);
    assert_eq!(
        complex.wrapping_pow(f64::MAX.into()),
        complex.pow(f64::MAX.into())
    );

    assert_eq!(
        Number::from(4i8).try_pow(0.5f64.into()).unwrap(),
        Number::from(2f64)
    );
    let input = Number::from(16_777_217i32);
    let exponent = Number::from(-1f64);
    assert_eq!(
        input.try_pow(exponent).unwrap(),
        Number::from(16_777_217f64).try_pow(exponent).unwrap()
    );
    assert_eq!(
        input.pow(exponent),
        Number::from(16_777_217f64).pow(exponent)
    );
    assert_eq!(input.wrapping_pow(exponent), input.pow(exponent));
}

#[test]
fn portable_error_and_complex_axis_contracts() {
    kind(Float::from(-104f32).try_exp(), ErrorKind::Underflow);
    kind(Float::from(-746f64).try_exp(), ErrorKind::Underflow);
    kind(Float::from(89f32).try_exp(), ErrorKind::Overflow);
    kind(Float::from(710f64).try_exp(), ErrorKind::Overflow);
    kind(Float::from(-1f64).try_ln(), ErrorKind::Domain);
    kind(
        Float::from(1f64).try_div(0f64.into()),
        ErrorKind::DivisionByZero,
    );
    kind(
        FloatType::F32.try_cast(f64::MAX.into()),
        ErrorKind::Overflow,
    );
    kind(
        FloatType::F32.try_cast(f64::MIN_POSITIVE.into()),
        ErrorKind::Underflow,
    );
    let z = Complex::from([1f64, 0.5]);
    macro_rules! unsupported {
        ($($checked:ident, $ordinary:ident;)*) => {$(
            kind(z.$checked(), ErrorKind::Unsupported);
            let native = z.$ordinary();
            assert!(f64::from(native.re()).is_finite());
            assert!(f64::from(native.im()).is_finite());
        )*};
    }
    unsupported! {
        try_exp, exp; try_ln, ln;
        try_sin, sin; try_cos, cos; try_tan, tan;
        try_sinh, sinh; try_cosh, cosh; try_tanh, tanh;
        try_asin, asin; try_acos, acos; try_atan, atan;
        try_asinh, asinh; try_acosh, acosh; try_atanh, atanh;
    }
    kind(
        Complex::from([1e300f64, 1.]).try_exp(),
        ErrorKind::Unsupported,
    );
    kind(
        Complex::from([f64::INFINITY, 0.]).try_exp(),
        ErrorKind::Unsupported,
    );
    kind(
        Complex::from([f64::NAN, 0.]).try_ln(),
        ErrorKind::Unsupported,
    );
    kind(z.try_pow(0.5f64.into()), ErrorKind::Unsupported);
    kind(
        Complex::from([0f64, 0.]).try_pow((-0.5f64).into()),
        ErrorKind::DivisionByZero,
    );
    kind(Complex::from([0f64, 1.]).try_atan(), ErrorKind::Domain);
    kind(Complex::from([1f64, 0.]).try_atanh(), ErrorKind::Domain);
    assert_eq!(z.try_pow(0f64.into()).unwrap(), Complex::from([1f64, 0.]));
    assert_eq!(
        Complex::from([-1f64, 0.]).try_pow(u64::MAX.into()).unwrap(),
        Complex::from([-1f64, 0.])
    );
    assert_eq!(
        Complex::from([3f64, 4.]).try_abs().unwrap(),
        Float::from(5f64)
    );
    for sign in [0.0f64, -0.0] {
        let value = Complex::from([-1f64, sign]).try_ln().unwrap();
        assert_eq!(f64::from(value.re()).to_bits(), 0f64.to_bits());
        assert_eq!(
            f64::from(value.im()).to_bits(),
            std::f64::consts::PI.copysign(sign).to_bits()
        );
        let value = Complex::from([sign, 1f64]).try_ln().unwrap();
        assert_eq!(f64::from(value.re()).to_bits(), 0f64.to_bits());
        assert_eq!(
            f64::from(value.im()).to_bits(),
            std::f64::consts::FRAC_PI_2.to_bits()
        );
        let value = Complex::from([2f64, sign])
            .try_log(Float::from(0.5f64))
            .unwrap();
        assert_eq!(f64::from(value.re()), -1.0);
        assert_eq!(
            f64::from(value.im()).is_sign_negative(),
            !sign.is_sign_negative()
        );
    }
    assert_eq!(
        Complex::from([0f64, 0.25]).try_sin().unwrap().im(),
        Float::from(0.25f64).try_sinh().unwrap()
    );
    assert_eq!(
        Complex::from([0f64, 0.25]).try_asinh().unwrap().im(),
        Float::from(0.25f64).try_asin().unwrap()
    );
}

#[test]
fn portable_complex_component_bits() {
    let result = Complex::from([0.25f32, -0.0]).try_exp().unwrap();
    let (Float::F32(re), Float::F32(im)) = (result.re(), result.im()) else {
        panic!("C32 dtype")
    };
    assert_eq!(re.to_bits(), 0x3fa45af2);
    assert_eq!(im.to_bits(), 0x80000000);
    let result = Complex::from([0f32, 0.25]).try_sin().unwrap();
    let (Float::F32(re), Float::F32(im)) = (result.re(), result.im()) else {
        panic!("C32 dtype")
    };
    assert_eq!(re.to_bits(), 0);
    assert_eq!(im.to_bits(), 0x3e815667);
    let result = Complex::from([-1f32, -0.0]).try_ln().unwrap();
    let (Float::F32(re), Float::F32(im)) = (result.re(), result.im()) else {
        panic!("C32 dtype")
    };
    assert_eq!(re.to_bits(), 0);
    assert_eq!(im.to_bits(), 0xc0490fdb);
    assert_eq!(
        Complex::from([3f32, 4.]).try_abs().unwrap().class(),
        FloatType::F32
    );
}
