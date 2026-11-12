//! Provides a generic [`Number`] enum with [`Boolean`], [`Complex`], [`Float`], [`Int`],
//! and [`UInt`] variants, as well as a [`NumberCollator`], [`ComplexCollator`], and
//! [`FloatCollator`] since these types do not implement [`Ord`].
//!
//! Checked numerical operations on [`NumberInstance`] and [`Trigonometry`] return
//! classified [`Error`]s, including overflow and underflow. Ordinary floating-point
//! operations use IEEE behavior; ordinary integers follow Rust overflow settings.
//! Mixed signed/unsigned ordinary arithmetic checks the selected destination.
//! [`Number::wrapping_add`] and the other `wrapping_*` methods make integer
//! overflow behavior explicit while retaining numeric promotion. Explicit casts through [`safecast`],
//! [`NumberClass::cast`], and [`NumberInstance::into_type`] remain lossy.
//!
//! Built-in checked real functions use pinned software math. Finite checked
//! complex transcendentals support certified axis identities; other inputs return
//! [`ErrorKind::Unsupported`] before evaluation. Portability requires the normal
//! IEEE environment. NaN payloads are unspecified; pointer-sized conversions
//! remain target-sized.
//!
//! For (de)serialization with `serde`, enable the `"serde"` feature.
//!
//! For (de)coding with `destream`, enable the `"stream"` feature.
//!
//! Example usage:
//! ```
//! # use number_general::{Int, Number};
//! # use safecast::CastFrom;
//! let sequence: Vec<Number> = vec![true.into(), 2.into(), 3.5.into(), [1.0, -0.5].into()];
//! let actual = sequence.into_iter().product();
//! let expected = Number::from(num::Complex::<f64>::new(7., -3.5));
//!
//! assert_eq!(expected, actual);
//! assert_eq!(Int::cast_from(actual), Int::from(7));
//! ```
//!
//! Choose the contract explicitly:
//! ```
//! use number_general::{ErrorKind, Number, NumberInstance};
//!
//! let max = Number::from(i8::MAX);
//! let one = Number::from(1i8);
//! assert_eq!(max.try_add(one).unwrap_err().kind(), ErrorKind::Overflow);
//! assert_eq!(max.wrapping_add(one), Number::from(i8::MIN));
//! assert!(f64::from(number_general::Float::from(1.) / 0.0.into()).is_infinite());
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use std::str::FromStr;

use collate::Collate;
use get_size::GetSize;
use safecast::{CastFrom, CastInto};

mod checked;
mod class;
#[cfg(feature = "stream")]
mod destream;
#[cfg(feature = "hash")]
mod hash;
mod instance;
mod native;
#[cfg(feature = "serde")]
mod serde;
mod wrapping;

pub use class::*;
pub use instance::*;

#[cfg(any(feature = "serde", feature = "stream"))]
const ERR_COMPLEX: &str = "a complex number";

#[cfg(any(feature = "serde", feature = "stream"))]
const ERR_NUMBER: &str = "a Number, like 1 or -2 or 3.14 or [0., -1.414]";

#[inline]
fn normalized_f64_bits(f: f64) -> u64 {
    if f == 0.0 {
        0
    } else if f.is_nan() {
        f64::NAN.to_bits()
    } else {
        f.to_bits()
    }
}

// Exactly 2^64, beyond the magnitude of every supported integer representation.
const INTEGER_BOUND_EXCLUSIVE: f64 = (u64::MAX as u128 + 1) as f64;

// Integer representations fit exactly in i128. Floating comparisons truncate only
// after checking their range, and compare the remaining fraction separately.
fn cmp_float_integer(float: f64, integer: i128) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float >= INTEGER_BOUND_EXCLUSIVE {
        return Some(Ordering::Greater);
    }
    if float <= -INTEGER_BOUND_EXCLUSIVE {
        return Some(Ordering::Less);
    }

    match (float as i128).cmp(&integer) {
        Ordering::Equal => float.fract().partial_cmp(&0.0),
        order => Some(order),
    }
}

/// Define a [`NumberType`] for a non-[`Number`] type such as a Rust primitive.
///
/// To access the `NumberType` of a `Number`, use `Instance::class`, e.g. `Number::from(1).class()`.
pub trait DType {
    fn dtype() -> NumberType;
}

macro_rules! dtype {
    ($t:ty, $nt:expr) => {
        impl DType for $t {
            fn dtype() -> NumberType {
                $nt
            }
        }
    };
}

dtype!(bool, NumberType::Bool);
dtype!(u8, NumberType::UInt(UIntType::U8));
dtype!(u16, NumberType::UInt(UIntType::U16));
dtype!(u32, NumberType::UInt(UIntType::U32));
dtype!(u64, NumberType::UInt(UIntType::U64));
dtype!(i8, NumberType::Int(IntType::I8));
dtype!(i16, NumberType::Int(IntType::I16));
dtype!(i32, NumberType::Int(IntType::I32));
dtype!(i64, NumberType::Int(IntType::I64));
dtype!(f32, NumberType::Float(FloatType::F32));
dtype!(f64, NumberType::Float(FloatType::F64));
dtype!(_Complex<f32>, NumberType::Complex(ComplexType::C32));
dtype!(_Complex<f64>, NumberType::Complex(ComplexType::C64));

/// A recoverable numeric failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    InvalidInput,
    Overflow,
    Underflow,
    DivisionByZero,
    Domain,
    Unrepresentable,
    Unsupported,
}

/// The error type returned when a numeric operation fails recoverably.
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    /// Construct an ordinary input error, preserving its diagnostic.
    pub fn new<E: fmt::Display>(cause: E) -> Self {
        Self::with_kind(ErrorKind::InvalidInput, cause)
    }

    /// Construct a classified error for a checked numeric implementation.
    pub fn with_kind<E: fmt::Display>(kind: ErrorKind, cause: E) -> Self {
        Self {
            kind,
            message: cause.to_string(),
        }
    }

    /// Inspect the failure without parsing diagnostic text.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl std::error::Error for Error {}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.message)
    }
}

type _Complex<T> = num::complex::Complex<T>;

/// A generic number.
#[derive(Clone, Copy, Eq, get_size_derive::GetSize)]
pub enum Number {
    Bool(Boolean),
    Complex(Complex),
    Float(Float),
    Int(Int),
    UInt(UInt),
}

impl Number {
    /// Return `true` if this is floating-point [`Number`], `false` if this is an integer.
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Complex(_) | Self::Float(_))
    }

    /// Return `true` if this is an [`Int`] or [`UInt`].
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_) | Self::UInt(_))
    }

    /// Return `false` if this is a [`Complex`] `Number`.
    pub fn is_real(&self) -> bool {
        !matches!(self, Self::Complex(_))
    }
}

macro_rules! number_binop {
    ($lhs:expr, $rhs:expr, $method:ident, $signed:ident, $unsigned:ident) => {{
        let left = $lhs;
        let right = $rhs;
        match checked::promote(left, right) {
            NumberType::Bool => Boolean::cast_from(left)
                .$method(Boolean::cast_from(right))
                .map(Number::from),
            NumberType::Int(dtype) => {
                checked::$signed(left.integer(), right.integer(), dtype).map(Number::from)
            }
            NumberType::UInt(dtype) => {
                checked::$unsigned(left.integer() as u128, right.integer() as u128, dtype)
                    .map(Number::from)
            }
            dtype => match (checked::cast(left, dtype)?, checked::cast(right, dtype)?) {
                (Number::Float(l), Number::Float(r)) => l.$method(r).map(Number::from),
                (Number::Complex(l), Number::Complex(r)) => l.$method(r).map(Number::from),
                _ => unreachable!("promoted family"),
            },
        }
    }};
}

macro_rules! ordinary_number_op {
    ($left:expr, $right:expr, $op:tt, $try:ident) => {{
        let left = $left;
        let right = $right;
        if matches!((left, right), (Number::Int(_), Number::UInt(_)) | (Number::UInt(_), Number::Int(_))) {
            left.$try(right).expect("mixed integer arithmetic")
        } else {
            let dtype = checked::promote(left, right);
            let left = checked::cast(left, dtype).expect("promotion");
            let right = checked::cast(right, dtype).expect("promotion");
            match (left, right) {
                (Number::Bool(l), Number::Bool(r)) => Number::from(l $op r),
                (Number::Int(l), Number::Int(r)) => Number::from(l $op r),
                (Number::UInt(l), Number::UInt(r)) => Number::from(l $op r),
                (Number::Float(l), Number::Float(r)) => Number::from(l $op r),
                (Number::Complex(l), Number::Complex(r)) => Number::from(l $op r),
                _ => unreachable!("promoted family"),
            }
        }
    }};
}

macro_rules! number_methods {
    ($($name:ident => $out:ident),*) => {$(
        fn $name(self) -> Self::$out {
            match self {
                Self::Bool(v) => v.$name().into(),
                Self::Int(v) => v.$name().into(),
                Self::UInt(v) => v.$name().into(),
                Self::Float(v) => v.$name().into(),
                Self::Complex(v) => v.$name().into(),
            }
        }
    )*};
}

macro_rules! checked_number_methods {
    ($($name:ident => $out:ident),*) => {$(
        fn $name(self) -> Result<Self::$out, Error> {
            match self {
                Self::Bool(v) => v.$name().map(Self::from),
                Self::Int(v) => v.$name().map(Self::from),
                Self::UInt(v) => v.$name().map(Self::from),
                Self::Float(v) => v.$name().map(Self::from),
                Self::Complex(v) => v.$name().map(Self::from),
            }
        }
    )*};
}

impl NumberInstance for Number {
    checked_number_methods!(try_abs => Abs, try_exp => Exp, try_ln => Log, try_round => Round);

    number_methods!(abs => Abs, exp => Exp, ln => Log, round => Round);
    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        let base = base.into();
        if matches!(self, Self::Complex(_)) || matches!(base, Self::Complex(_)) {
            let dtype = checked::promote(self, base);
            let Self::Complex(value) = checked::cast(self, dtype).expect("complex promotion")
            else {
                unreachable!("complex")
            };
            value.log(base).into()
        } else {
            crate::native::float::log(self, base).into()
        }
    }

    fn pow(self, exponent: Number) -> Self {
        if matches!(self, Self::Complex(_)) || matches!(exponent, Self::Complex(_)) {
            let dtype = match checked::promote(self, exponent) {
                NumberType::Complex(d) => d,
                _ => unreachable!("complex"),
            };
            let Self::Complex(value) =
                checked::cast(self, dtype.into()).expect("complex promotion")
            else {
                unreachable!("complex")
            };
            return value.pow(exponent).into();
        }
        if native::promotes_power(exponent) {
            let Self::Float(value) =
                checked::cast(self, native::power_float_type(self, exponent).into())
                    .expect("float promotion")
            else {
                unreachable!("float")
            };
            return value.pow(exponent).into();
        }
        match self {
            Self::Bool(v) => v.pow(exponent).into(),
            Self::Int(v) => v.pow(exponent).into(),
            Self::UInt(v) => v.pow(exponent).into(),
            Self::Float(v) => v.pow(exponent).into(),
            Self::Complex(_) => unreachable!("complex branch"),
        }
    }

    type Abs = Number;
    type Exp = Self;
    type Log = Self;
    type Round = Self;
    type Class = NumberType;

    fn class(&self) -> NumberType {
        match self {
            Self::Bool(_) => NumberType::Bool,
            Self::Complex(c) => c.class().into(),
            Self::Float(f) => f.class().into(),
            Self::Int(i) => i.class().into(),
            Self::UInt(u) => u.class().into(),
        }
    }

    fn into_type(self, dtype: NumberType) -> Number {
        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let b: Boolean = self.cast_into();
                b.into()
            }
            NT::Complex(ct) => {
                let c: Complex = self.cast_into();
                c.into_type(ct).into()
            }
            NT::Float(ft) => {
                let f: Float = self.cast_into();
                f.into_type(ft).into()
            }
            NT::Int(it) => {
                let i: Int = self.cast_into();
                i.into_type(it).into()
            }
            NT::UInt(ut) => {
                let u: UInt = self.cast_into();
                u.into_type(ut).into()
            }
            NT::Number => self,
        }
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        number_binop!(self, other, try_add, add_signed, add_unsigned)
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        number_binop!(self, other, try_sub, sub_signed, sub_unsigned)
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        number_binop!(self, other, try_mul, mul_signed, mul_unsigned)
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        number_binop!(self, other, try_div, div_signed, div_unsigned)
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        number_binop!(self, other, try_rem, rem_signed, rem_unsigned)
    }

    fn try_into_type(self, dtype: NumberType) -> Result<Self, Error> {
        checked::into_type(self, dtype)
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        let base = base.into();
        match self {
            Self::Complex(value) => value.try_log(base).map(Self::from),
            value => checked::log(value, base).map(Self::from),
        }
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        match self {
            Self::Complex(value) => value.try_pow(exponent).map(Self::from),
            Self::Float(value) => value.try_pow(exponent).map(Self::from),
            value if native::promotes_power(exponent) => {
                let Self::Float(value) =
                    checked::cast(value, native::power_float_type(value, exponent).into())?
                else {
                    unreachable!("float conversion")
                };
                value.try_pow(exponent).map(Self::from)
            }
            Self::Bool(value) => value.try_pow(exponent).map(Self::from),
            Self::Int(value) => value.try_pow(exponent).map(Self::from),
            Self::UInt(value) => value.try_pow(exponent).map(Self::from),
        }
    }
}

impl FloatInstance for Number {
    fn is_infinite(&self) -> bool {
        match self {
            Self::Complex(c) => c.is_infinite(),
            Self::Float(f) => f.is_infinite(),
            _ => false,
        }
    }

    fn is_nan(&self) -> bool {
        match self {
            Self::Complex(c) => c.is_nan(),
            Self::Float(f) => f.is_nan(),
            _ => false,
        }
    }
}

impl Number {
    fn components(self) -> (Self, f64) {
        if let Self::Complex(c) = self {
            let c: num::Complex<f64> = c.into();
            (Self::Float(c.re.into()), c.im)
        } else {
            (self, 0.0)
        }
    }

    fn integer(self) -> i128 {
        match self {
            Self::Bool(b) => i128::from(bool::from(b)),
            Self::Int(i) => i64::from(i) as i128,
            Self::UInt(u) => u64::from(u) as i128,
            _ => unreachable!("real integer comparison requires an integer"),
        }
    }

    fn cmp_real(self, other: Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Float(l), Self::Float(r)) => l.partial_cmp(&r),
            (Self::Float(l), r) => cmp_float_integer(f64::from(l), r.integer()),
            (l, Self::Float(r)) => {
                cmp_float_integer(f64::from(r), l.integer()).map(Ordering::reverse)
            }
            (l, r) => Some(l.integer().cmp(&r.integer())),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        let (l, li) = self.components();
        let (r, ri) = other.components();
        normalized_f64_bits(li) == normalized_f64_bits(ri) && l.cmp_real(r) == Some(Ordering::Equal)
    }
}

impl Hash for Number {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (real, imaginary) = self.components();
        if let Self::Float(f) = real {
            let f = f64::from(f);
            if f > -INTEGER_BOUND_EXCLUSIVE && f < INTEGER_BOUND_EXCLUSIVE && f.fract() == 0.0 {
                0u8.hash(state);
                (f as i128).hash(state);
            } else {
                1u8.hash(state);
                normalized_f64_bits(f).hash(state);
            }
        } else {
            0u8.hash(state);
            real.integer().hash(state);
        }
        normalized_f64_bits(imaginary).hash(state);
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        let (l, li) = self.components();
        let (r, ri) = other.components();
        if li == 0.0 && ri == 0.0 {
            l.cmp_real(r)
        } else {
            None
        }
    }
}

impl Add for Number {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ordinary_number_op!(self, other, +, try_add)
    }
}

impl AddAssign for Number {
    fn add_assign(&mut self, other: Self) {
        let sum = *self + other;
        *self = sum;
    }
}

impl Rem for Number {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        ordinary_number_op!(self, other, %, try_rem)
    }
}

impl RemAssign for Number {
    fn rem_assign(&mut self, other: Self) {
        let rem = *self % other;
        *self = rem;
    }
}

impl Sub for Number {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ordinary_number_op!(self, other, -, try_sub)
    }
}

impl SubAssign for Number {
    fn sub_assign(&mut self, other: Self) {
        let difference = *self - other;
        *self = difference;
    }
}

impl Sum for Number {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            return Self::default().class().zero();
        };
        iter.fold(first, |a, b| a + b)
    }
}

impl Mul for Number {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        ordinary_number_op!(self, other, *, try_mul)
    }
}

impl MulAssign for Number {
    fn mul_assign(&mut self, other: Self) {
        let product = *self * other;
        *self = product;
    }
}

impl Div for Number {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        ordinary_number_op!(self, other, /, try_div)
    }
}

impl DivAssign for Number {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Number {
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(first) = iter.next() else {
            return Self::default().class().one();
        };
        iter.fold(first, |a, b| a * b)
    }
}

impl Trigonometry for Number {
    checked_number_methods!(try_asin => Out, try_sin => Out, try_sinh => Out, try_asinh => Out, try_acos => Out, try_cos => Out, try_cosh => Out, try_acosh => Out, try_atan => Out, try_tan => Out, try_tanh => Out, try_atanh => Out);

    number_methods!(asin => Out, sin => Out, sinh => Out, asinh => Out, acos => Out, cos => Out, cosh => Out, acosh => Out, atan => Out, tan => Out, tanh => Out, atanh => Out);

    type Out = Self;
}

impl Default for Number {
    fn default() -> Self {
        Self::Bool(Boolean::default())
    }
}

impl From<bool> for Number {
    fn from(b: bool) -> Self {
        Self::Bool(b.into())
    }
}

impl From<Boolean> for Number {
    fn from(b: Boolean) -> Number {
        Number::Bool(b)
    }
}

macro_rules! from_uint {
    ($($t:ty),+ $(,)?) => {
        $(
            impl From<$t> for Number {
                fn from(u: $t) -> Self {
                    Self::UInt(u.into())
                }
            }
        )+
    };
}

macro_rules! from_int {
    ($($t:ty),+ $(,)?) => {
        $(
            impl From<$t> for Number {
                fn from(i: $t) -> Self {
                    Self::Int(i.into())
                }
            }
        )+
    };
}

macro_rules! from_float {
    ($($t:ty),+ $(,)?) => {
        $(
            impl From<$t> for Number {
                fn from(f: $t) -> Self {
                    Self::Float(f.into())
                }
            }
        )+
    };
}

from_uint!(u8, u16, u32, u64);

impl From<UInt> for Number {
    fn from(u: UInt) -> Number {
        Number::UInt(u)
    }
}

from_int!(i8, i16, i32, i64);

impl From<Int> for Number {
    fn from(i: Int) -> Number {
        Number::Int(i)
    }
}

from_float!(f32, f64);

impl From<Float> for Number {
    fn from(f: Float) -> Number {
        Number::Float(f)
    }
}

impl From<[f32; 2]> for Number {
    fn from(arr: [f32; 2]) -> Self {
        Self::Complex(arr.into())
    }
}

impl From<[f64; 2]> for Number {
    fn from(arr: [f64; 2]) -> Self {
        Self::Complex(arr.into())
    }
}

impl From<_Complex<f32>> for Number {
    fn from(c: _Complex<f32>) -> Self {
        Self::Complex(c.into())
    }
}

impl From<_Complex<f64>> for Number {
    fn from(c: _Complex<f64>) -> Self {
        Self::Complex(c.into())
    }
}

impl From<Complex> for Number {
    fn from(c: Complex) -> Number {
        Number::Complex(c)
    }
}

impl FromStr for Number {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();

        if s.is_empty() {
            Err(Error::new(
                "cannot parse a Number from an empty string".to_string(),
            ))
        } else {
            Int::from_str(s)
                .map(Self::from)
                .or_else(|_| UInt::from_str(s).map(Self::from))
                .or_else(|_| Float::from_str(s).map(Self::from))
                .or_else(|_| Complex::from_str(s).map(Self::from))
                .or_else(|_| Boolean::from_str(s).map(Self::from))
        }
    }
}

impl CastFrom<Number> for Boolean {
    fn cast_from(number: Number) -> Boolean {
        match number {
            Number::Bool(b) => b,
            Number::Complex(c) => Boolean::cast_from(c),
            Number::Float(f) => Boolean::cast_from(f),
            Number::Int(i) => Boolean::cast_from(i),
            Number::UInt(u) => Boolean::from(bool::cast_from(u)),
        }
    }
}

impl CastFrom<Number> for bool {
    fn cast_from(n: Number) -> bool {
        Boolean::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for _Complex<f32> {
    fn cast_from(n: Number) -> _Complex<f32> {
        Complex::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for _Complex<f64> {
    fn cast_from(n: Number) -> _Complex<f64> {
        Complex::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for f32 {
    fn cast_from(n: Number) -> f32 {
        Float::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for f64 {
    fn cast_from(n: Number) -> f64 {
        Float::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for Float {
    fn cast_from(number: Number) -> Float {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => f,
            Int(i) => Self::cast_from(i),
            UInt(u) => Self::cast_from(u),
        }
    }
}

impl CastFrom<Number> for i8 {
    fn cast_from(n: Number) -> i8 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i16 {
    fn cast_from(n: Number) -> i16 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i32 {
    fn cast_from(n: Number) -> i32 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i64 {
    fn cast_from(n: Number) -> i64 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for Int {
    fn cast_from(number: Number) -> Int {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => Self::cast_from(f),
            Int(i) => i,
            UInt(u) => Self::cast_from(u),
        }
    }
}

impl CastFrom<Number> for u8 {
    fn cast_from(n: Number) -> u8 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u16 {
    fn cast_from(n: Number) -> u16 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u32 {
    fn cast_from(n: Number) -> u32 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u64 {
    fn cast_from(n: Number) -> u64 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for usize {
    fn cast_from(n: Number) -> usize {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for UInt {
    fn cast_from(number: Number) -> UInt {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => Self::cast_from(f),
            Int(i) => Self::cast_from(i),
            UInt(u) => u,
        }
    }
}

/// Defines a collation order for [`Number`].
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct NumberCollator {
    complex: ComplexCollator,
}

impl Collate for NumberCollator {
    type Value = Number;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        let (l, li) = left.components();
        let (r, ri) = right.components();
        match (li == 0.0, ri == 0.0) {
            (true, true) => l.cmp_real(r).unwrap_or_else(|| {
                if l.is_nan() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }),
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self
                .complex
                .cmp(&Complex::cast_from(*left), &Complex::cast_from(*right)),
        }
    }
}

/// A struct for deserializing a `Number` which implements
/// [`destream::de::Visitor`] and [`serde::de::Visitor`].
#[cfg(any(feature = "serde", feature = "stream"))]
pub struct NumberVisitor;

#[cfg(any(feature = "serde", feature = "stream"))]
impl NumberVisitor {
    #[inline]
    fn bool<E>(self, b: bool) -> Result<Number, E> {
        Ok(Number::Bool(b.into()))
    }

    #[inline]
    fn i8<E>(self, i: i8) -> Result<Number, E> {
        Ok(Number::Int(Int::I8(i)))
    }

    #[inline]
    fn i16<E>(self, i: i16) -> Result<Number, E> {
        Ok(Number::Int(Int::I16(i)))
    }

    #[inline]
    fn i32<E>(self, i: i32) -> Result<Number, E> {
        Ok(Number::Int(Int::I32(i)))
    }

    #[inline]
    fn i64<E>(self, i: i64) -> Result<Number, E> {
        Ok(Number::Int(Int::I64(i)))
    }

    #[inline]
    fn u8<E>(self, u: u8) -> Result<Number, E> {
        Ok(Number::UInt(UInt::U8(u)))
    }

    #[inline]
    fn u16<E>(self, u: u16) -> Result<Number, E> {
        Ok(Number::UInt(UInt::U16(u)))
    }

    #[inline]
    fn u32<E>(self, u: u32) -> Result<Number, E> {
        Ok(Number::UInt(UInt::U32(u)))
    }

    #[inline]
    fn u64<E>(self, u: u64) -> Result<Number, E> {
        Ok(Number::UInt(UInt::U64(u)))
    }

    #[inline]
    fn f32<E>(self, f: f32) -> Result<Number, E> {
        Ok(Number::Float(Float::F32(f)))
    }

    #[inline]
    fn f64<E>(self, f: f64) -> Result<Number, E> {
        Ok(Number::Float(Float::F64(f)))
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({})", self, self.class())
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Bool(b) => fmt::Display::fmt(b, f),
            Number::Complex(c) => fmt::Display::fmt(c, f),
            Number::Float(n) => fmt::Display::fmt(n, f),
            Number::Int(i) => fmt::Display::fmt(i, f),
            Number::UInt(u) => fmt::Display::fmt(u, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::approx_constant)] // tests use simple decimal literals for readability
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
        assert!((-n).ln().is_nan());
        assert_eq!(
            Number::from(-n).try_ln().unwrap_err().kind(),
            ErrorKind::Domain
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
                assert_eq!(zero, two % one);
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
    #[allow(clippy::approx_constant)] // tests use explicit literals to validate parsing
    fn test_parse() {
        for value in [
            Number::from(u64::MAX),
            Number::from(i64::MIN),
            Number::from(true),
            Number::from(false),
        ] {
            assert_eq!(Number::from_str(&value.to_string()).unwrap(), value);
        }
        assert!(matches!(Number::from_str("12").unwrap(), Number::Int(_)));
        assert!(matches!(
            Number::from_str(&u64::MAX.to_string()).unwrap(),
            Number::UInt(_)
        ));
        for (text, value) in [
            ("1e+2+3i", num::Complex::new(100., 3.)),
            ("1e-2+3i", num::Complex::new(0.01, 3.)),
            ("1+2e-3i", num::Complex::new(1., 0.002)),
            ("3i", num::Complex::new(0., 3.)),
            (" -2i ", num::Complex::new(0., -2.)),
        ] {
            assert_eq!(Number::from_str(text).unwrap(), Number::from(value));
        }

        for invalid in [
            "",
            " ",
            "not-a-number",
            "1e",
            "1+xi",
            "18446744073709551616x",
        ] {
            assert!(Number::from_str(invalid).is_err(), "{invalid}");
        }
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
}

#[cfg(test)]
#[path = "../tests/unit/numeric_contract.rs"]
mod numeric_contract;

#[cfg(test)]
#[path = "../tests/unit/checked.rs"]
mod checked_contract;

#[cfg(test)]
#[path = "../tests/unit/contracts.rs"]
mod numerical_contracts;
