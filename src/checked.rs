//! Shared checked conversions and result validation. No host or backend policy.

use num::{BigRational, Signed, Zero};

use crate::{
    Boolean, Complex, ComplexType, Error, ErrorKind, Float, FloatInstance, FloatType, Int, IntType,
    Number, NumberInstance, NumberType, UInt, UIntType,
};

pub(super) mod math;

type Result<T> = std::result::Result<T, Error>;

pub(super) fn error(kind: ErrorKind, operation: &str) -> Error {
    Error::with_kind(kind, format!("{operation}: {kind:?}"))
}

pub(super) fn float_type(value: Number) -> FloatType {
    match value {
        Number::Float(value) => value.class(),
        Number::Complex(Complex::C64(_))
        | Number::Int(Int::I64(_))
        | Number::UInt(UInt::U64(_)) => FloatType::F64,
        _ => FloatType::F32,
    }
}

pub(super) fn signed_type(value: Number) -> IntType {
    match value {
        Number::Int(value) => value.class(),
        Number::Bool(_) => IntType::I16,
        Number::UInt(UInt::U8(_) | UInt::U16(_)) => IntType::I16,
        Number::UInt(UInt::U32(_))
        | Number::Float(Float::F32(_))
        | Number::Complex(Complex::C32(_)) => IntType::I32,
        _ => IntType::I64,
    }
}

fn unsigned_type(value: Number) -> UIntType {
    match value {
        Number::UInt(value) => value.class(),
        Number::Bool(_) | Number::Int(Int::I8(_)) => UIntType::U8,
        Number::Int(Int::I16(_)) => UIntType::U16,
        Number::Int(Int::I32(_))
        | Number::Float(Float::F32(_))
        | Number::Complex(Complex::C32(_)) => UIntType::U32,
        _ => UIntType::U64,
    }
}

pub(super) fn target(value: Number, dtype: NumberType) -> NumberType {
    match dtype {
        NumberType::Int(IntType::Int) => NumberType::Int(signed_type(value)),
        NumberType::UInt(UIntType::UInt) => NumberType::UInt(unsigned_type(value)),
        NumberType::Float(FloatType::Float) => NumberType::Float(float_type(value)),
        NumberType::Complex(ComplexType::Complex) => {
            NumberType::Complex(if float_type(value) == FloatType::F64 {
                ComplexType::C64
            } else {
                ComplexType::C32
            })
        }
        other => other,
    }
}

pub(super) fn promote(left: Number, right: Number) -> NumberType {
    match Ord::max(left.class(), right.class()) {
        NumberType::Int(_) => NumberType::Int(Ord::max(signed_type(left), signed_type(right))),
        NumberType::UInt(_) => {
            NumberType::UInt(Ord::max(unsigned_type(left), unsigned_type(right)))
        }
        NumberType::Float(_) => NumberType::Float(Ord::max(float_type(left), float_type(right))),
        NumberType::Complex(_) => NumberType::Complex(
            if Ord::max(float_type(left), float_type(right)) == FloatType::F64 {
                ComplexType::C64
            } else {
                ComplexType::C32
            },
        ),
        other => other,
    }
}

pub(super) fn into_type(value: Number, dtype: NumberType) -> Result<Number> {
    cast(value, target(value, dtype))
}

fn real(value: Number) -> Result<Number> {
    match value {
        Number::Complex(value) => {
            if f64::from(value.im()) != 0.0 {
                Err(error(
                    ErrorKind::Unrepresentable,
                    "discarding an imaginary component",
                ))
            } else {
                Ok(value.re().into())
            }
        }
        other => Ok(other),
    }
}

pub(super) fn signed(value: i128, dtype: IntType) -> Result<Int> {
    macro_rules! narrow {
        ($t:ty, $variant:ident) => {{
            if value < <$t>::MIN as i128 {
                return Err(error(ErrorKind::Underflow, "integer result"));
            }
            if value > <$t>::MAX as i128 {
                return Err(error(ErrorKind::Overflow, "integer result"));
            }
            Ok(Int::$variant(value as $t))
        }};
    }
    match dtype {
        IntType::I8 => narrow!(i8, I8),
        IntType::I16 => narrow!(i16, I16),
        IntType::I32 => narrow!(i32, I32),
        IntType::I64 => narrow!(i64, I64),
        IntType::Int => unreachable!("resolved integer width"),
    }
}

pub(super) fn unsigned(value: u128, dtype: UIntType) -> Result<UInt> {
    macro_rules! narrow {
        ($t:ty, $variant:ident) => {{
            if value > <$t>::MAX as u128 {
                return Err(error(ErrorKind::Overflow, "integer result"));
            }
            Ok(UInt::$variant(value as $t))
        }};
    }
    match dtype {
        UIntType::U8 => narrow!(u8, U8),
        UIntType::U16 => narrow!(u16, U16),
        UIntType::U32 => narrow!(u32, U32),
        UIntType::U64 => narrow!(u64, U64),
        UIntType::UInt => unreachable!("resolved integer width"),
    }
}

fn signed_bounds(dtype: IntType) -> (i128, i128) {
    match dtype {
        IntType::I8 => (i8::MIN.into(), i8::MAX.into()),
        IntType::I16 => (i16::MIN.into(), i16::MAX.into()),
        IntType::I32 => (i32::MIN.into(), i32::MAX.into()),
        IntType::I64 => (i64::MIN.into(), i64::MAX.into()),
        IntType::Int => unreachable!("resolved integer width"),
    }
}

fn unsigned_max(dtype: UIntType) -> u128 {
    match dtype {
        UIntType::U8 => u8::MAX.into(),
        UIntType::U16 => u16::MAX.into(),
        UIntType::U32 => u32::MAX.into(),
        UIntType::U64 => u64::MAX.into(),
        UIntType::UInt => unreachable!("resolved integer width"),
    }
}

fn cast_signed(value: Number, dtype: IntType) -> Result<Int> {
    let value = match real(value)? {
        Number::Float(value) => {
            let value = f64::from(value);
            let (min, max) = signed_bounds(dtype);
            if !value.is_finite() {
                return Err(error(
                    ErrorKind::Unrepresentable,
                    "nonfinite integer conversion",
                ));
            }
            if value < min as f64 {
                return Err(error(ErrorKind::Underflow, "integer conversion"));
            }
            if value >= (max + 1) as f64 {
                return Err(error(ErrorKind::Overflow, "integer conversion"));
            }
            if value.fract() != 0.0 {
                return Err(error(
                    ErrorKind::Unrepresentable,
                    "fractional integer conversion",
                ));
            }
            value as i128
        }
        other => other.integer(),
    };
    signed(value, dtype)
}

fn cast_unsigned(value: Number, dtype: UIntType) -> Result<UInt> {
    let value = match real(value)? {
        Number::Float(value) => {
            let value = f64::from(value);
            if !value.is_finite() {
                return Err(error(
                    ErrorKind::Unrepresentable,
                    "nonfinite integer conversion",
                ));
            }
            if value < 0.0 {
                return Err(error(ErrorKind::Underflow, "unsigned conversion"));
            }
            if value >= (unsigned_max(dtype) + 1) as f64 {
                return Err(error(ErrorKind::Overflow, "unsigned conversion"));
            }
            if value.fract() != 0.0 {
                return Err(error(
                    ErrorKind::Unrepresentable,
                    "fractional integer conversion",
                ));
            }
            value as u128
        }
        other => {
            let value = other.integer();
            if value < 0 {
                return Err(error(ErrorKind::Underflow, "unsigned conversion"));
            }
            value as u128
        }
    };
    unsigned(value, dtype)
}

// Select both widths before conversion, so a wider base never follows a lossy
// intermediate conversion of the value.
pub(super) fn real_arguments(value: Number, base: Number) -> Result<(Float, Float)> {
    let dtype = Ord::max(float_type(value), float_type(base));
    match (cast(value, dtype.into())?, cast(base, dtype.into())?) {
        (Number::Float(value), Number::Float(base)) => Ok((value, base)),
        _ => unreachable!("real logarithm arguments"),
    }
}

pub(super) fn log(value: Number, base: Number) -> Result<Float> {
    if matches!(base, Number::Complex(_)) {
        return Err(error(
            ErrorKind::Unsupported,
            "checked complex logarithm base",
        ));
    }
    let (value, base) = real_arguments(value, base)?;
    let finite = finite(value) && finite(base);
    if finite && (f64::from(value) <= 0.0 || f64::from(base) <= 0.0 || f64::from(base) == 1.0) {
        return Err(error(ErrorKind::Domain, "log"));
    }
    float_result(
        finite,
        math::log(value, base),
        f64::from(value) == 1.0,
        ErrorKind::Underflow,
        "log",
    )
}

fn cast_float(value: Number, dtype: FloatType) -> Result<Float> {
    let value = real(value)?;
    let source = match value {
        Number::Float(f) => f64::from(f),
        other => other.integer() as f64,
    };
    match dtype {
        FloatType::F64 => Ok(Float::F64(source)),
        FloatType::F32 => {
            if source.is_finite() && source.abs() > f64::from(f32::MAX) {
                return Err(error(ErrorKind::Overflow, "float conversion"));
            }
            let result = match value {
                Number::Int(i) => i64::from(i) as f32,
                Number::UInt(u) => u64::from(u) as f32,
                _ => source as f32,
            };
            if source != 0.0 && result == 0.0 {
                return Err(error(ErrorKind::Underflow, "float conversion"));
            }
            Ok(Float::F32(result))
        }
        FloatType::Float => unreachable!("resolved floating width"),
    }
}

pub(super) fn cast(value: Number, dtype: NumberType) -> Result<Number> {
    match dtype {
        NumberType::Number => Ok(value),
        NumberType::Bool => {
            let value = real(value)?;
            if value == Number::from(0) {
                Ok(Boolean::from(false).into())
            } else if value == Number::from(1) {
                Ok(Boolean::from(true).into())
            } else {
                Err(error(ErrorKind::Unrepresentable, "Boolean conversion"))
            }
        }
        NumberType::Int(dtype) => cast_signed(value, dtype).map(Number::from),
        NumberType::UInt(dtype) => cast_unsigned(value, dtype).map(Number::from),
        NumberType::Float(dtype) => cast_float(value, dtype).map(Number::from),
        NumberType::Complex(dtype) => {
            let (re, im) = match value {
                Number::Complex(c) => (Number::from(c.re()), Number::from(c.im())),
                other => (other, Number::from(0)),
            };
            match dtype {
                ComplexType::C32 => {
                    let re = cast_float(re, FloatType::F32)?;
                    let im = cast_float(im, FloatType::F32)?;
                    Ok(Complex::C32(num::Complex::new(
                        f64::from(re) as f32,
                        f64::from(im) as f32,
                    ))
                    .into())
                }
                ComplexType::C64 => Ok(Complex::C64(num::Complex::new(
                    f64::from(cast_float(re, FloatType::F64)?),
                    f64::from(cast_float(im, FloatType::F64)?),
                ))
                .into()),
                ComplexType::Complex => unreachable!("resolved complex width"),
            }
        }
    }
}

pub(super) fn finite(value: Float) -> bool {
    !value.is_nan() && !value.is_infinite()
}

pub(super) fn complex_finite(value: Complex) -> bool {
    finite(value.re()) && finite(value.im())
}

pub(super) fn float_result(
    finite_inputs: bool,
    result: Float,
    zero_valid: bool,
    lost_zero: ErrorKind,
    operation: &str,
) -> Result<Float> {
    if finite_inputs {
        if result.is_nan() {
            return Err(error(ErrorKind::Domain, operation));
        }
        if result.is_infinite() {
            return Err(error(ErrorKind::Overflow, operation));
        }
        if f64::from(result) == 0.0 && !zero_valid {
            return Err(error(lost_zero, operation));
        }
    }
    Ok(result)
}

// Only zero/nonfinite complex components require exact, bounded cancellation work.
// Rational inputs are finite binary floats; their sizes are bounded by the dtype.
pub(super) fn complex_binary(
    left: Complex,
    right: Complex,
    result: Complex,
    exact: impl FnOnce(BigRational, BigRational, BigRational, BigRational) -> (BigRational, BigRational),
    operation: &str,
) -> Result<Complex> {
    if !complex_finite(left) || !complex_finite(right) {
        return Ok(result);
    }
    let values = [f64::from(result.re()), f64::from(result.im())];
    if values.iter().all(|x| x.is_finite() && *x != 0.0) {
        return Ok(result);
    }
    let rational = |x: Float| BigRational::from_float(f64::from(x)).expect("finite component");
    let (re, im) = exact(
        rational(left.re()),
        rational(left.im()),
        rational(right.re()),
        rational(right.im()),
    );
    let (max, min) = match result {
        Complex::C32(_) => (
            f64::from(f32::MAX),
            f64::from(f32::MIN_POSITIVE * f32::EPSILON),
        ),
        Complex::C64(_) => (f64::MAX, f64::MIN_POSITIVE * f64::EPSILON),
    };
    let max = BigRational::from_float(max).expect("finite maximum");
    let min = BigRational::from_float(min).expect("finite minimum subnormal");
    for (value, exact) in values.into_iter().zip([re, im]) {
        if !value.is_finite() {
            return Err(error(
                if exact.abs() > max {
                    ErrorKind::Overflow
                } else {
                    ErrorKind::Unsupported
                },
                operation,
            ));
        }
        if value == 0.0 && !exact.is_zero() {
            return Err(error(
                if exact.abs() * BigRational::from_integer(2.into()) <= min {
                    ErrorKind::Underflow
                } else {
                    ErrorKind::Unsupported
                },
                operation,
            ));
        }
    }
    Ok(result)
}

// Inputs are supported 64-bit values. Their widened mathematical results fit
// i128/u128; only unsigned subtraction and destination narrowing can fail.
pub(super) fn add_signed(left: i128, right: i128, dtype: IntType) -> Result<Int> {
    signed(left + right, dtype)
}

pub(super) fn sub_signed(left: i128, right: i128, dtype: IntType) -> Result<Int> {
    signed(left - right, dtype)
}

pub(super) fn mul_signed(left: i128, right: i128, dtype: IntType) -> Result<Int> {
    signed(left * right, dtype)
}

pub(super) fn div_signed(left: i128, right: i128, dtype: IntType) -> Result<Int> {
    if right == 0 {
        return Err(error(ErrorKind::DivisionByZero, "div"));
    }
    signed(left / right, dtype)
}

pub(super) fn rem_signed(left: i128, right: i128, dtype: IntType) -> Result<Int> {
    if right == 0 {
        return Err(error(ErrorKind::DivisionByZero, "rem"));
    }
    signed(left % right, dtype)
}

pub(super) fn add_unsigned(left: u128, right: u128, dtype: UIntType) -> Result<UInt> {
    unsigned(left + right, dtype)
}

pub(super) fn sub_unsigned(left: u128, right: u128, dtype: UIntType) -> Result<UInt> {
    unsigned(
        left.checked_sub(right)
            .ok_or_else(|| error(ErrorKind::Underflow, "sub"))?,
        dtype,
    )
}

pub(super) fn mul_unsigned(left: u128, right: u128, dtype: UIntType) -> Result<UInt> {
    unsigned(left * right, dtype)
}

pub(super) fn div_unsigned(left: u128, right: u128, dtype: UIntType) -> Result<UInt> {
    if right == 0 {
        return Err(error(ErrorKind::DivisionByZero, "div"));
    }
    unsigned(left / right, dtype)
}

pub(super) fn rem_unsigned(left: u128, right: u128, dtype: UIntType) -> Result<UInt> {
    if right == 0 {
        return Err(error(ErrorKind::DivisionByZero, "rem"));
    }
    unsigned(left % right, dtype)
}

pub(super) fn pow_signed(base: Int, exponent: Number) -> Result<Int> {
    let exponent = crate::native::integer_exponent(exponent)?;
    if exponent.negative {
        return match i64::from(base) {
            0 => Err(error(ErrorKind::DivisionByZero, "pow")),
            1 => Ok(base),
            -1 => signed(
                if exponent.magnitude & 1 == 0 { 1 } else { -1 },
                base.class(),
            ),
            _ => Err(error(
                ErrorKind::Unrepresentable,
                "fractional integer power",
            )),
        };
    }
    let negative = i64::from(base) < 0 && exponent.magnitude & 1 != 0;
    crate::native::power(
        base,
        exponent.magnitude,
        signed(1, base.class())?,
        Int::try_mul,
    )
    .map_err(|_| {
        error(
            if negative {
                ErrorKind::Underflow
            } else {
                ErrorKind::Overflow
            },
            "pow",
        )
    })
}

pub(super) fn pow_unsigned(base: UInt, exponent: Number) -> Result<UInt> {
    let exponent = crate::native::integer_exponent(exponent)?;
    if exponent.negative {
        return match u64::from(base) {
            0 => Err(error(ErrorKind::DivisionByZero, "pow")),
            1 => Ok(base),
            _ => Err(error(
                ErrorKind::Unrepresentable,
                "fractional integer power",
            )),
        };
    }
    crate::native::power(
        base,
        exponent.magnitude,
        unsigned(1, base.class())?,
        UInt::try_mul,
    )
}
