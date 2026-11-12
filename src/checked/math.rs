//! Pinned software kernels for checked real calculations and complex axis signs.

use crate::Float;

macro_rules! unary {
    ($($name:ident: $f32:ident, $f64:ident;)*) => {$(
        pub(crate) fn $name(value: Float) -> Float {
            match value {
                Float::F32(value) => Float::F32(libm::$f32(value)),
                Float::F64(value) => Float::F64(libm::$f64(value)),
            }
        }
    )*};
}

unary! {
    abs: fabsf, fabs;
    round: roundf, round;

    exp: expf, exp;
    ln: logf, log;

    sin: sinf, sin;
    cos: cosf, cos;
    tan: tanf, tan;
    sinh: sinhf, sinh;
    cosh: coshf, cosh;
    tanh: tanhf, tanh;

    asin: asinf, asin;
    acos: acosf, acos;
    atan: atanf, atan;
    asinh: asinhf, asinh;
    acosh: acoshf, acosh;
    atanh: atanhf, atanh;
}

macro_rules! binary {
    ($($name:ident: $f32:ident, $f64:ident;)*) => {$(
        pub(crate) fn $name(left: Float, right: Float) -> Float {
            match (left, right) {
                (Float::F32(left), Float::F32(right)) => Float::F32(libm::$f32(left, right)),
                (Float::F64(left), Float::F64(right)) => Float::F64(libm::$f64(left, right)),
                _ => unreachable!("prepared floating widths"),
            }
        }
    )*};
}

binary! {
    pow: powf, pow;
    hypot: hypotf, hypot;
    atan2: atan2f, atan2;
}

pub(crate) fn log(value: Float, base: Float) -> Float {
    ln(value) / ln(base)
}

// The input is an axis zero. Only the coefficient's sign is relevant; avoid
// manufacturing NaN when its unused magnitude overflows.
pub(crate) fn axis_product(zero: Float, coefficient: Float) -> Float {
    match (zero, coefficient) {
        (Float::F32(zero), Float::F32(coefficient)) => {
            Float::F32(zero * libm::copysignf(1.0, coefficient))
        }
        (Float::F64(zero), Float::F64(coefficient)) => {
            Float::F64(zero * libm::copysign(1.0, coefficient))
        }
        _ => unreachable!("axis component widths"),
    }
}

// Share the analytic identities, but use portable kernels even for auxiliary signs.
use super::math as float;
crate::native::axis_functions!(float);
