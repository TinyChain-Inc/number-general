//! Provides a generic [`Number`] enum with [`Boolean`], [`Complex`], [`Float`], [`Int`],
//! and [`UInt`] variants, as well as a [`NumberCollator`], [`ComplexCollator`], and
//! [`FloatCollator`] since these types do not implement [`Ord`].
//!
//! `Number` supports casting with [`safecast`].
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
#![expect(
    clippy::derived_hash_with_manual_eq,
    reason = "TODO: fix this or add an explanation"
)]

use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::*;
use std::str::FromStr;

use collate::{Collate, Collator};
use get_size::GetSize;
use get_size_derive::*;
use safecast::{CastFrom, CastInto};

mod class;
#[cfg(feature = "stream")]
mod destream;
#[cfg(feature = "hash")]
mod hash;
mod instance;
#[cfg(feature = "serde")]
mod serde;

#[cfg(test)]
mod tests;

pub use class::*;
pub use instance::*;

#[cfg(any(feature = "serde", feature = "stream"))]
const ERR_COMPLEX: &str = "a complex number";

#[cfg(any(feature = "serde", feature = "stream"))]
const ERR_NUMBER: &str = "a Number, like 1 or -2 or 3.14 or [0., -1.414]";

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

/// The error type returned when a `Number` operation fails recoverably.
pub struct Error(String);

impl Error {
    fn new<E: fmt::Display>(cause: E) -> Self {
        Self(cause.to_string())
    }
}

impl std::error::Error for Error {}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

type _Complex<T> = num::complex::Complex<T>;

/// A generic number.
#[derive(Clone, Copy, Eq, Hash, GetSize)]
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

impl NumberInstance for Number {
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

    fn abs(self) -> Number {
        use Number::*;
        match self {
            Complex(c) => Float(c.abs()),
            Float(f) => Float(f.abs()),
            Int(i) => Int(i.abs()),
            other => other,
        }
    }

    fn exp(self) -> Self::Exp {
        match self {
            Self::Complex(this) => this.exp().into(),
            Self::Float(this) => this.exp().into(),
            this => Float::cast_from(this).exp().into(),
        }
    }

    fn ln(self) -> Self::Log {
        match self {
            Self::Bool(b) => b.ln().into(),
            Self::Complex(this) => this.ln().into(),
            Self::Float(this) => this.ln().into(),
            Self::Int(this) => this.ln().into(),
            Self::UInt(this) => this.ln().into(),
        }
    }

    fn log<N: NumberInstance>(self, base: N) -> Self::Log
    where
        Float: From<N>,
    {
        match self {
            Self::Complex(this) => this.log(base).into(),
            Self::Float(this) => this.log(base).into(),
            this => Float::cast_from(this).log(base).into(),
        }
    }

    fn pow(self, exp: Self) -> Self {
        match self {
            Self::Complex(this) => Self::Complex(this.pow(exp)),
            Self::Float(this) => Self::Float(this.pow(exp)),
            Self::Int(this) => match exp {
                Self::Complex(exp) => Self::Float(Float::from(this).pow(exp.into())),
                Self::Float(exp) => Self::Float(Float::from(this).pow(exp.into())),
                Self::Int(exp) if exp < exp.class().zero() => {
                    Self::Float(Float::from(this).pow(exp.into()))
                }
                exp => Self::Int(this.pow(exp)),
            },
            Self::UInt(this) => match exp {
                Self::Complex(exp) => Self::Float(Float::from(this).pow(exp.into())),
                Self::Float(exp) => Self::Float(Float::from(this).pow(exp.into())),
                Self::Int(exp) if exp < exp.class().zero() => {
                    Self::Float(Float::from(this).pow(exp.into()))
                }
                exp => Self::UInt(this.pow(exp)),
            },
            Self::Bool(b) => Self::Bool(b.pow(exp)),
        }
    }

    fn round(self) -> Self::Round {
        match self {
            Self::Complex(c) => c.round().into(),
            Self::Float(f) => f.round().into(),
            other => other,
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

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l), Self::Int(r)) => l.eq(r),
            (Self::UInt(l), Self::UInt(r)) => l.eq(r),
            (Self::Float(l), Self::Float(r)) => l.eq(r),
            (Self::Bool(l), Self::Bool(r)) => l.eq(r),
            (Self::Complex(l), Self::Complex(r)) => l.eq(r),

            (Self::Complex(l), r) => l.eq(&Complex::cast_from(*r)),
            (Self::Float(l), r) => l.eq(&Float::cast_from(*r)),
            (Self::Int(l), r) => l.eq(&Int::cast_from(*r)),
            (Self::UInt(l), r) => l.eq(&UInt::cast_from(*r)),

            (l, r) => r.eq(l),
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(l), Self::Int(r)) => l.partial_cmp(r),
            (Self::UInt(l), Self::UInt(r)) => l.partial_cmp(r),
            (Self::Float(l), Self::Float(r)) => l.partial_cmp(r),
            (Self::Bool(l), Self::Bool(r)) => l.partial_cmp(r),
            (Self::Complex(_), _) => None,
            (_, Self::Complex(_)) => None,

            (l, r) => {
                let dtype = Ord::max(l.class(), r.class());
                let l = l.into_type(dtype);
                let r = r.into_type(dtype);
                l.partial_cmp(&r)
            }
        }
    }
}

impl Add for Number {
    type Output = Self;

    fn add(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
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

    fn rem(self, other: Self) -> Self::Output {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this % other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this % other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this % other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this % other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this % other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
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

    fn sub(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
    }
}

impl SubAssign for Number {
    fn sub_assign(&mut self, other: Self) {
        let difference = *self - other;
        *self = difference;
    }
}

impl Sum for Number {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = NumberType::Number.zero();
        for i in iter {
            sum += i;
        }
        sum
    }
}

impl Mul for Number {
    type Output = Self;

    fn mul(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
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

    fn div(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Number => panic!("A number instance must have a specific type, not Number"),
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this / other.cast_into()).into()
            }
        }
    }
}

impl DivAssign for Number {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Number {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = NumberType::Number.zero();
        let mut product = NumberType::Number.one();

        for i in iter {
            if i == zero {
                return zero;
            }

            product *= i;
        }
        product
    }
}

macro_rules! trig {
    ($fun:ident) => {
        fn $fun(self) -> Self::Out {
            match self {
                Self::Bool(b) => b.$fun().into(),
                Self::Complex(c) => c.$fun().into(),
                Self::Float(f) => f.$fun().into(),
                Self::Int(i) => i.$fun().into(),
                Self::UInt(u) => u.$fun().into(),
            }
        }
    };
}

impl Trigonometry for Number {
    type Out = Self;

    trig! {asin}
    trig! {sin}
    trig! {asinh}
    trig! {sinh}

    trig! {acos}
    trig! {cos}
    trig! {acosh}
    trig! {cosh}

    trig! {atan}
    trig! {tan}
    trig! {atanh}
    trig! {tanh}
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

impl From<u8> for Number {
    fn from(u: u8) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u16> for Number {
    fn from(u: u16) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u32> for Number {
    fn from(u: u32) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u64> for Number {
    fn from(u: u64) -> Self {
        Self::UInt(u.into())
    }
}

impl From<UInt> for Number {
    fn from(u: UInt) -> Number {
        Number::UInt(u)
    }
}

impl From<i8> for Number {
    fn from(i: i8) -> Self {
        Self::Int(i.into())
    }
}

impl From<i16> for Number {
    fn from(i: i16) -> Self {
        Self::Int(i.into())
    }
}

impl From<i32> for Number {
    fn from(i: i32) -> Self {
        Self::Int(i.into())
    }
}

impl From<i64> for Number {
    fn from(i: i64) -> Self {
        Self::Int(i.into())
    }
}

impl From<Int> for Number {
    fn from(i: Int) -> Number {
        Number::Int(i)
    }
}

impl From<f32> for Number {
    fn from(f: f32) -> Self {
        Self::Float(f.into())
    }
}

impl From<f64> for Number {
    fn from(f: f64) -> Self {
        Self::Float(f.into())
    }
}

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
            Err(Error(
                "cannot parse a Number from an empty string".to_string(),
            ))
        } else if s.len() == 1 {
            Int::from_str(s).map(Self::from)
        } else if s.contains("e+") || s.contains("e-") {
            Float::from_str(s).map(Self::from)
        } else if s[1..].contains('+') || s[1..].contains('-') {
            Complex::from_str(s).map(Self::from)
        } else if s.contains('.') || s.contains('e') {
            Float::from_str(s).map(Self::from)
        } else {
            Int::from_str(s).map(Self::from)
        }
    }
}

impl CastFrom<Number> for Boolean {
    fn cast_from(number: Number) -> Boolean {
        if number == number.class().zero() {
            false.into()
        } else {
            true.into()
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
    bool: Collator<Boolean>,
    complex: ComplexCollator,
    float: FloatCollator,
    int: Collator<Int>,
    uint: Collator<UInt>,
}

impl Collate for NumberCollator {
    type Value = Number;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        match (left, right) {
            (Number::Bool(l), Number::Bool(r)) => self.bool.cmp(l, r),
            (Number::Complex(l), Number::Complex(r)) => self.complex.cmp(l, r),
            (Number::Float(l), Number::Float(r)) => self.float.cmp(l, r),
            (Number::Int(l), Number::Int(r)) => self.int.cmp(l, r),
            (Number::UInt(l), Number::UInt(r)) => self.uint.cmp(l, r),
            (l, r) => {
                let dtype = Ord::max(l.class(), r.class());
                let l = l.into_type(dtype);
                let r = r.into_type(dtype);
                self.cmp(&l, &r)
            }
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
        Ok(Number::Int(Int::I16(i as i16)))
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self, self.class())
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Bool(b) => fmt::Display::fmt(b, f),
            Number::Complex(c) => fmt::Display::fmt(c, f),
            Number::Float(n) => fmt::Display::fmt(n, f),
            Number::Int(i) => fmt::Display::fmt(i, f),
            Number::UInt(u) => fmt::Display::fmt(u, f),
        }
    }
}
