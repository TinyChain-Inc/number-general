//! Primitive numerical work shared by native and checked scalar operations.

use crate::{
    checked, Complex, Error, ErrorKind, Float, Int, Number, NumberClass, NumberInstance, UInt,
};

#[derive(Clone, Copy)]
pub(crate) struct Exponent {
    pub negative: bool,
    pub magnitude: u64,
}

pub(crate) fn integer_exponent(value: Number) -> Result<Exponent, Error> {
    let (negative, magnitude) = match value {
        Number::Bool(value) => (false, u64::from(bool::from(value))),
        Number::Int(value) => {
            let value = i64::from(value);
            (value < 0, value.unsigned_abs())
        }
        Number::UInt(value) => (false, u64::from(value)),
        Number::Float(value) => {
            let value = f64::from(value);

            if !value.is_finite() || value.abs() >= crate::INTEGER_BOUND_EXCLUSIVE {
                return Err(checked::error(
                    ErrorKind::Unsupported,
                    "integer exponent range",
                ));
            }

            if value.fract() != 0.0 {
                return Err(checked::error(
                    ErrorKind::Unrepresentable,
                    "fractional integer exponent",
                ));
            }

            (value < 0.0, value.abs() as u64)
        }
        Number::Complex(_) => {
            return Err(checked::error(ErrorKind::Unsupported, "complex exponent"))
        }
    };

    Ok(Exponent {
        negative,
        magnitude,
    })
}

// Only fractional, negative, nonfinite or complex exponents require dynamic
// floating/complex evaluation. Positive integral floats retain integer bases.
pub(crate) fn promotes_power(exponent: Number) -> bool {
    match exponent {
        Number::Float(value) => {
            let value = f64::from(value);
            !value.is_finite() || value < 0.0 || value.fract() != 0.0
        }
        Number::Int(value) => i64::from(value) < 0,
        Number::Complex(_) => true,
        _ => false,
    }
}

pub(crate) fn power_float_type(base: Number, exponent: Number) -> crate::FloatType {
    let dtype = checked::float_type(base);
    match exponent {
        Number::Float(value) => Ord::max(dtype, value.class()),
        _ => dtype,
    }
}

pub(crate) fn power<T: Copy>(
    mut base: T,
    mut exponent: u64,
    mut result: T,
    mul: impl Fn(T, T) -> Result<T, Error>,
) -> Result<T, Error> {
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul(result, base)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = mul(base, base)?;
        }
    }
    Ok(result)
}

pub(crate) fn int_pow(base: Int, exponent: Number) -> Int {
    let exp = integer_exponent(exponent).expect("invalid integer exponent");
    if exp.negative {
        return checked::pow_signed(base, exponent).expect("invalid negative integer power");
    }
    power(base, exp.magnitude, base.class().one(), |a, b| Ok(a * b)).expect("primitive power")
}

pub(crate) fn uint_pow(base: UInt, exponent: Number) -> UInt {
    let exp = integer_exponent(exponent).expect("invalid integer exponent");
    if exp.negative {
        return checked::pow_unsigned(base, exponent).expect("invalid negative integer power");
    }
    power(base, exp.magnitude, base.class().one(), |a, b| Ok(a * b)).expect("primitive power")
}

pub(crate) fn parts(re: Float, im: Float) -> Complex {
    match (re, im) {
        (Float::F32(re), Float::F32(im)) => Complex::from([re, im]),
        (Float::F64(re), Float::F64(im)) => Complex::from([re, im]),
        _ => unreachable!("component widths match"),
    }
}

pub(crate) fn negate(value: Float) -> Float {
    match value {
        Float::F32(v) => Float::F32(-v),
        Float::F64(v) => Float::F64(-v),
    }
}

// A finite real axis has an established real-domain result. The callback chooses
// ordinary or checked evaluation; zero components follow the analytic identity.
macro_rules! real_axis {
    ($($name:ident, $domain:expr, $imag:expr;)*) => {$ (
        pub(crate) fn $name(value: $crate::Complex, eval: impl FnOnce($crate::Float) -> Result<$crate::Float, $crate::Error>) -> Result<Option<$crate::Complex>, $crate::Error> {
            let re = value.re();
            let im = value.im();
            let x = f64::from(re);
            if $crate::checked::complex_finite(value) && f64::from(im) == 0.0 && ($domain)(x) {
                let result = eval(re)?;
                let im = ($imag)(re, im);
                return Ok(Some($crate::native::parts(result, im)));
            }
            Ok(None)
        }
    )*};
}

pub(crate) use real_axis;

macro_rules! axis_functions {
    ($math:ident) => {
        $crate::native::real_axis! {
            exp_axis, |_| true, |_: Float, im: Float| im;
            ln_axis, |x| x > 0.0, |_: Float, im: Float| im;
            sin_axis, |_| true, |re: Float, im: Float| $math::axis_product(im, $math::cos(re));
            cos_axis, |_| true, |re: Float, im: Float| $crate::native::negate($math::axis_product(im, $math::sin(re)));
            tan_axis, |_| true, |_: Float, im: Float| im;
            sinh_axis, |_| true, |re: Float, im: Float| $math::axis_product(im, $math::cosh(re));
            cosh_axis, |_| true, |re: Float, im: Float| $math::axis_product(im, $math::sinh(re));
            tanh_axis, |_| true, |_: Float, im: Float| im;
            asin_axis, |x: f64| x.abs() <= 1.0, |_: Float, im: Float| im;
            acos_axis, |x: f64| x.abs() <= 1.0, |_: Float, im: Float| $crate::native::negate(im);
            atan_axis, |_| true, |_: Float, im: Float| im;
            asinh_axis, |_| true, |_: Float, im: Float| im;
            acosh_axis, |x| x >= 1.0, |_: Float, im: Float| im;
            atanh_axis, |x: f64| x.abs() < 1.0, |_: Float, im: Float| im;
        }
    };
}

pub(crate) use axis_functions;
axis_functions!(float);

pub(crate) mod float {
    use super::{integer_exponent, power};
    use crate::{checked, Float, FloatType, Number, NumberClass, NumberInstance};

    macro_rules! unary {
        ($($name:ident),*) => {$(
            pub(crate) fn $name(value: Float) -> Float {
                match value {
                    Float::F32(v) => Float::F32(v.$name()),
                    Float::F64(v) => Float::F64(v.$name()),
                }
            }
        )*};
    }

    unary!(
        abs, exp, ln, round, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh
    );

    pub(crate) fn axis_product(zero: Float, coefficient: Float) -> Float {
        zero * coefficient
    }

    pub(crate) fn log(value: Number, base: Number) -> Float {
        let (value, base) = checked::real_arguments(value, base).expect("real logarithm arguments");
        match (value, base) {
            (Float::F32(value), Float::F32(base)) => Float::F32(value.log(base)),
            (Float::F64(value), Float::F64(base)) => Float::F64(value.log(base)),
            _ => unreachable!("prepared floating widths"),
        }
    }

    pub(crate) fn pow(value: Float, exponent: Number) -> Float {
        if !matches!(exponent, Number::Float(_) | Number::Complex(_)) {
            let exp = integer_exponent(exponent).expect("integer input");
            let one = value.class().one();
            let base = if exp.negative { one / value } else { value };
            return power(base, exp.magnitude, one, |a, b| Ok(a * b)).expect("primitive power");
        }
        let dtype = if matches!(exponent, Number::Float(Float::F64(_))) {
            FloatType::F64
        } else {
            value.class()
        };
        let value = value.into_type(dtype);
        let Number::Float(exp) = checked::cast(exponent, dtype.into()).expect("real exponent")
        else {
            unreachable!("float")
        };
        match (value, exp) {
            (Float::F32(v), Float::F32(e)) => Float::F32(v.powf(e)),
            (Float::F64(v), Float::F64(e)) => Float::F64(v.powf(e)),
            _ => unreachable!("promoted widths"),
        }
    }
}

pub(crate) mod complex {
    use super::{float, integer_exponent, power};
    use crate::{checked, Complex, ComplexType, Float, Number, NumberClass, NumberInstance};

    macro_rules! unary {
        ($($name:ident, $axis:ident;)*) => {$(
            pub(crate) fn $name(value: Complex) -> Complex {
                if let Some(result) = super::$axis(value, |v| Ok(float::$name(v))).expect("primitive axis") {
                    return result;
                }
                match value {
                    Complex::C32(v) => Complex::C32(v.$name()),
                    Complex::C64(v) => Complex::C64(v.$name()),
                }
            }
        )*};
    }

    unary! {
        exp, exp_axis; ln, ln_axis;
        sin, sin_axis; cos, cos_axis; tan, tan_axis;
        sinh, sinh_axis; cosh, cosh_axis; tanh, tanh_axis;
        asin, asin_axis; acos, acos_axis; atan, atan_axis;
        asinh, asinh_axis; acosh, acosh_axis; atanh, atanh_axis;
    }

    pub(crate) fn abs(value: Complex) -> Float {
        match value {
            Complex::C32(v) => Float::F32(v.norm()),
            Complex::C64(v) => Float::F64(v.norm()),
        }
    }

    pub(crate) fn round(value: Complex) -> Complex {
        match value {
            Complex::C32(v) => Complex::from([v.re.round(), v.im.round()]),
            Complex::C64(v) => Complex::from([v.re.round(), v.im.round()]),
        }
    }

    pub(crate) fn log(value: Complex, base: Number) -> Complex {
        let dtype = checked::promote(value.into(), base);
        let dtype = match dtype {
            crate::NumberType::Complex(dtype) => dtype,
            _ => unreachable!("complex family"),
        };
        let value = value.into_type(dtype);
        let Number::Complex(base) = checked::cast(base, dtype.into()).expect("complex base") else {
            unreachable!("complex")
        };
        ln(value) / ln(base)
    }

    pub(crate) fn pow(value: Complex, exponent: Number) -> Complex {
        if !matches!(exponent, Number::Float(_) | Number::Complex(_)) {
            let exp = integer_exponent(exponent).expect("integer input");
            let one = value.class().one();
            let base = if exp.negative { one / value } else { value };
            return power(base, exp.magnitude, one, |a, b| Ok(a * b)).expect("primitive power");
        }
        let dtype = match exponent {
            Number::Float(Float::F64(_)) | Number::Complex(Complex::C64(_)) => ComplexType::C64,
            _ => value.class(),
        };
        let value = value.into_type(dtype);
        if matches!(exponent, Number::Complex(_)) {
            let Number::Complex(exp) =
                checked::cast(exponent, dtype.into()).expect("complex exponent")
            else {
                unreachable!("complex")
            };
            match (value, exp) {
                (Complex::C32(v), Complex::C32(e)) => Complex::C32(v.powc(e)),
                (Complex::C64(v), Complex::C64(e)) => Complex::C64(v.powc(e)),
                _ => unreachable!("promoted widths"),
            }
        } else {
            let Number::Float(exp) =
                checked::cast(exponent, value.re().class().into()).expect("real exponent")
            else {
                unreachable!("float")
            };
            match (value, exp) {
                (Complex::C32(v), Float::F32(e)) => Complex::C32(v.powf(e)),
                (Complex::C64(v), Float::F64(e)) => Complex::C64(v.powf(e)),
                _ => unreachable!("promoted widths"),
            }
        }
    }
}
