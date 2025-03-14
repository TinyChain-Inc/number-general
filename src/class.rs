use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::*;

use safecast::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::instance::{Boolean, Complex, Float, Int, UInt};
use super::{Number, _Complex};

/// Defines common properties of numeric types supported by [`Number`].
pub trait NumberClass: Default + Into<NumberType> + Ord + Send + fmt::Display {
    type Instance: NumberInstance;

    /// Cast the given `Number` into an instance of this type.
    fn cast(&self, n: Number) -> Self::Instance;

    /// Return `true` if this is a complex type.
    fn is_complex(&self) -> bool {
        false
    }

    /// Return `false` if this is a complex type.
    fn is_real(&self) -> bool {
        !self.is_complex()
    }

    /// Return the maximum size of this type of [`Number`], in bits.
    fn size(self) -> usize;

    /// Return `1` as an instance of this type.
    fn one(&self) -> <Self as NumberClass>::Instance;

    /// Return `0` as an instance of this type.
    fn zero(&self) -> <Self as NumberClass>::Instance;
}

/// Defines common operations on numeric types supported by [`Number`].
pub trait NumberInstance:
    Copy
    + Default
    + Sized
    + From<Boolean>
    + Into<Number>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Product
    + Sum
    + fmt::Debug
    + fmt::Display
{
    type Abs: NumberInstance;
    type Exp: NumberInstance;
    type Log: NumberInstance;
    type Round: NumberInstance;
    type Class: NumberClass<Instance = Self>;

    /// Get an impl of [`NumberClass`] describing this number.
    fn class(&self) -> Self::Class;

    /// Cast this number into the specified [`NumberClass`].
    fn into_type(
        self,
        dtype: <Self as NumberInstance>::Class,
    ) -> <<Self as NumberInstance>::Class as NumberClass>::Instance;

    /// Calculate the absolute value of this number.
    fn abs(self) -> Self::Abs;

    /// Raise `e` to the power of this number.
    fn exp(self) -> Self::Exp;

    /// Compute the natural logarithm of this number.
    fn ln(self) -> Self::Log;

    /// Compute the logarithm of this number with respect to the given `base`.
    fn log<N: NumberInstance>(self, base: N) -> Self::Log
    where
        Float: From<N>;

    /// Raise this number to the given exponent.
    ///
    /// Panics: if the given exponent is a complex number.
    fn pow(self, exp: Number) -> Self;

    /// Return `true` if `self` and `other` are nonzero.
    fn and(self, other: Self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        Boolean::cast_from(self)
            .and(Boolean::cast_from(other))
            .into()
    }

    /// Return `true` if this number is zero.
    fn not(self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        Boolean::cast_from(self).not().into()
    }

    /// Return `true` if `self` or `other` is nonzero.
    fn or(self, other: Self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        let this = Boolean::cast_from(self);
        let that = Boolean::cast_from(other);
        this.or(that).into()
    }

    /// Return this number rounded to the nearest integer.
    fn round(self) -> Self::Round;

    /// Return `true` if exactly one of `self` and `other` is zero.
    fn xor(self, other: Self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        let this = Boolean::cast_from(self);
        let that = Boolean::cast_from(other);
        this.xor(that).into()
    }
}

/// Trigonometric functions.
pub trait Trigonometry {
    type Out: NumberInstance;

    /// Arcsine
    fn asin(self) -> Self::Out;

    /// Sine
    fn sin(self) -> Self::Out;

    /// Hyperbolic arcsine
    fn asinh(self) -> Self::Out;

    /// Hyperbolic sine
    fn sinh(self) -> Self::Out;

    /// Hyperbolic arccosine
    fn acos(self) -> Self::Out;

    /// Cosine
    fn cos(self) -> Self::Out;

    /// Hyperbolic arccosine
    fn acosh(self) -> Self::Out;

    /// Hyperbolic cosine
    fn cosh(self) -> Self::Out;

    /// Arctangent
    fn atan(self) -> Self::Out;

    /// Tangent
    fn tan(self) -> Self::Out;

    /// Hyperbolic arctangent
    fn atanh(self) -> Self::Out;

    /// Hyperbolic tangent
    fn tanh(self) -> Self::Out;
}

/// Defines common operations on real (i.e. not `Complex`) numbers.
pub trait RealInstance: PartialEq + PartialOrd + Sized {
    const ONE: Self;
    const ZERO: Self;

    /// Return `true` if this is zero or a positive number.
    fn is_positive(&self) -> bool {
        self >= &Self::ZERO
    }

    /// Return `true` if this is a negative number.
    fn is_negative(&self) -> bool {
        self < &Self::ZERO
    }

    /// Return `true` if this is zero.
    fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }
}

/// Defines common operations on floating-point numeric types.
pub trait FloatInstance {
    /// Return `true` if this `Number` is infinite (e.g. [`f32::INFINITY`]).
    fn is_infinite(&self) -> bool;

    /// Return `true` if this is not a valid number (NaN).
    fn is_nan(&self) -> bool;
}

/// The type of a [`Complex`] number.
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplexType {
    C32,
    C64,
    Complex,
}

impl Default for ComplexType {
    fn default() -> Self {
        Self::Complex
    }
}

impl NumberClass for ComplexType {
    type Instance = Complex;

    fn cast(&self, n: Number) -> Complex {
        match self {
            Self::C64 => Complex::C64(n.cast_into()),
            _ => Complex::C32(n.cast_into()),
        }
    }

    fn is_complex(&self) -> bool {
        true
    }

    fn size(self) -> usize {
        match self {
            Self::C32 => 8,
            Self::C64 => 16,
            Self::Complex => 16,
        }
    }

    fn one(&self) -> Complex {
        match self {
            Self::C32 => _Complex::<f32>::new(1f32, 0f32).into(),
            Self::C64 => _Complex::<f64>::new(1f64, 0f64).into(),
            Self::Complex => _Complex::<f32>::new(1f32, 0f32).into(),
        }
    }

    fn zero(&self) -> Complex {
        match self {
            Self::C32 => _Complex::<f32>::new(0f32, 0f32).into(),
            Self::C64 => _Complex::<f64>::new(0f64, 0f64).into(),
            Self::Complex => _Complex::<f32>::new(0f32, 0f32).into(),
        }
    }
}

impl Ord for ComplexType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::C32, Self::C32) => Ordering::Equal,
            (Self::C64, Self::C64) => Ordering::Equal,
            (Self::Complex, Self::Complex) => Ordering::Equal,

            (Self::Complex, _) => Ordering::Greater,
            (_, Self::Complex) => Ordering::Less,

            (Self::C64, Self::C32) => Ordering::Greater,
            (Self::C32, Self::C64) => Ordering::Less,
        }
    }
}

impl PartialOrd for ComplexType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<ComplexType> for NumberType {
    fn from(ct: ComplexType) -> NumberType {
        Self::Complex(ct)
    }
}

impl fmt::Debug for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::C32 => write!(f, "32-bit complex number"),
            Self::C64 => write!(f, "64-bit complex number"),
            Self::Complex => write!(f, "complex number"),
        }
    }
}

/// The type of a [`Boolean`].
#[derive(Clone, Copy, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BooleanType;

impl NumberClass for BooleanType {
    type Instance = Boolean;

    fn cast(&self, n: Number) -> Boolean {
        n.cast_into()
    }

    fn size(self) -> usize {
        1
    }

    fn one(&self) -> Boolean {
        true.into()
    }

    fn zero(&self) -> Boolean {
        false.into()
    }
}

impl Ord for BooleanType {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl PartialOrd for BooleanType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<BooleanType> for NumberType {
    fn from(_bt: BooleanType) -> NumberType {
        NumberType::Bool
    }
}

impl fmt::Debug for BooleanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for BooleanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Boolean")
    }
}

/// The type of a [`Float`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FloatType {
    F32,
    F64,
    Float,
}

impl Default for FloatType {
    fn default() -> Self {
        Self::Float
    }
}

impl NumberClass for FloatType {
    type Instance = Float;

    fn cast(&self, n: Number) -> Float {
        match self {
            Self::F64 => Float::F64(n.cast_into()),
            _ => Float::F32(n.cast_into()),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Float => 8,
        }
    }

    fn one(&self) -> Float {
        match self {
            Self::F32 => 1f32.into(),
            Self::F64 => 1f64.into(),
            Self::Float => 1f32.into(),
        }
    }

    fn zero(&self) -> Float {
        match self {
            Self::F32 => 0f32.into(),
            Self::F64 => 0f64.into(),
            Self::Float => 0f32.into(),
        }
    }
}

impl Ord for FloatType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::F32, Self::F32) => Ordering::Equal,
            (Self::F64, Self::F64) => Ordering::Equal,
            (Self::Float, Self::Float) => Ordering::Equal,

            (Self::Float, _) => Ordering::Greater,
            (_, Self::Float) => Ordering::Less,

            (Self::F64, Self::F32) => Ordering::Greater,
            (Self::F32, Self::F64) => Ordering::Less,
        }
    }
}

impl PartialOrd for FloatType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<FloatType> for NumberType {
    fn from(ft: FloatType) -> NumberType {
        NumberType::Float(ft)
    }
}

impl fmt::Debug for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use FloatType::*;
        match self {
            F32 => write!(f, "32-bit float"),
            F64 => write!(f, "64-bit float"),
            Float => write!(f, "float"),
        }
    }
}

/// The type of an [`Int`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IntType {
    I8,
    I16,
    I32,
    I64,
    Int,
}

impl Default for IntType {
    fn default() -> Self {
        Self::Int
    }
}

impl NumberClass for IntType {
    type Instance = Int;

    fn cast(&self, n: Number) -> Int {
        match self {
            Self::I8 => Int::I8(n.cast_into()),
            Self::I16 => Int::I16(n.cast_into()),
            Self::Int | Self::I32 => Int::I32(n.cast_into()),
            Self::I64 => Int::I64(n.cast_into()),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Int => 8,
        }
    }

    fn one(&self) -> Int {
        match self {
            Self::I8 => 1i8.into(),
            Self::I16 => 1i16.into(),
            Self::I32 => 1i32.into(),
            Self::I64 => 1i64.into(),
            Self::Int => 1i16.into(),
        }
    }

    fn zero(&self) -> Int {
        match self {
            Self::I8 => 0i8.into(),
            Self::I16 => 0i16.into(),
            Self::I32 => 0i32.into(),
            Self::I64 => 0i64.into(),
            Self::Int => 0i16.into(),
        }
    }
}

impl Ord for IntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (this, that) if this == that => Ordering::Equal,

            (Self::Int, _) => Ordering::Greater,
            (_, Self::Int) => Ordering::Less,

            (Self::I64, _) => Ordering::Greater,
            (_, Self::I64) => Ordering::Less,

            (Self::I32, _) => Ordering::Greater,
            (_, Self::I32) => Ordering::Less,

            (Self::I16, _) => Ordering::Greater,
            (_, Self::I16) => Ordering::Less,

            (Self::I8, _) => Ordering::Less,
        }
    }
}

impl PartialOrd for IntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<IntType> for NumberType {
    fn from(it: IntType) -> NumberType {
        NumberType::Int(it)
    }
}

impl fmt::Debug for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::I8 => write!(f, "8-bit integer"),
            Self::I16 => write!(f, "16-bit integer"),
            Self::I32 => write!(f, "32-bit integer"),
            Self::I64 => write!(f, "64-bit integer"),
            Self::Int => write!(f, "integer"),
        }
    }
}

/// The type of a [`UInt`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UIntType {
    U8,
    U16,
    U32,
    U64,
    UInt,
}

impl Default for UIntType {
    fn default() -> Self {
        Self::UInt
    }
}

impl NumberClass for UIntType {
    type Instance = UInt;

    fn cast(&self, n: Number) -> UInt {
        match self {
            Self::U8 => UInt::U8(n.cast_into()),
            Self::U16 => UInt::U16(n.cast_into()),
            Self::UInt | Self::U32 => UInt::U32(n.cast_into()),
            Self::U64 => UInt::U64(n.cast_into()),
        }
    }

    fn size(self) -> usize {
        match self {
            UIntType::U8 => 1,
            UIntType::U16 => 2,
            UIntType::U32 => 4,
            UIntType::U64 => 8,
            UIntType::UInt => 8,
        }
    }

    fn one(&self) -> UInt {
        match self {
            Self::U8 => 1u8.into(),
            Self::U16 => 1u16.into(),
            Self::U32 => 1u32.into(),
            Self::U64 => 1u64.into(),
            Self::UInt => 1u8.into(),
        }
    }

    fn zero(&self) -> UInt {
        match self {
            Self::U8 => 0u8.into(),
            Self::U16 => 0u16.into(),
            Self::U32 => 0u32.into(),
            Self::U64 => 0u64.into(),
            Self::UInt => 0u8.into(),
        }
    }
}

impl Ord for UIntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::U8, Self::U8) => Ordering::Equal,
            (Self::U16, Self::U16) => Ordering::Equal,
            (Self::U32, Self::U32) => Ordering::Equal,
            (Self::U64, Self::U64) => Ordering::Equal,
            (Self::UInt, Self::UInt) => Ordering::Equal,

            (Self::UInt, _) => Ordering::Greater,
            (_, Self::UInt) => Ordering::Less,

            (Self::U64, _) => Ordering::Greater,
            (_, Self::U64) => Ordering::Less,

            (Self::U8, _) => Ordering::Less,
            (_, Self::U8) => Ordering::Greater,

            (Self::U32, Self::U16) => Ordering::Greater,
            (Self::U16, Self::U32) => Ordering::Less,
        }
    }
}

impl PartialOrd for UIntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<UIntType> for NumberType {
    fn from(ut: UIntType) -> NumberType {
        NumberType::UInt(ut)
    }
}

impl fmt::Debug for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use UIntType::*;
        match self {
            U8 => write!(f, "8-bit unsigned"),
            U16 => write!(f, "16-bit unsigned"),
            U32 => write!(f, "32-bit unsigned"),
            U64 => write!(f, "64-bit unsigned"),
            UInt => write!(f, "uint"),
        }
    }
}

/// The type of a generic [`Number`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NumberType {
    Bool,
    Complex(ComplexType),
    Float(FloatType),
    Int(IntType),
    UInt(UIntType),
    Number,
}

impl NumberType {
    pub fn uint64() -> Self {
        NumberType::UInt(UIntType::U64)
    }
}

impl Default for NumberType {
    fn default() -> Self {
        Self::Number
    }
}

impl NumberClass for NumberType {
    type Instance = Number;

    fn is_complex(&self) -> bool {
        match self {
            Self::Bool => BooleanType.is_complex(),
            Self::Complex(ct) => ct.is_complex(),
            Self::Float(ft) => ft.is_complex(),
            Self::Int(it) => it.is_complex(),
            Self::UInt(ut) => ut.is_complex(),
            Self::Number => false,
        }
    }

    fn cast(&self, n: Number) -> Number {
        match self {
            Self::Bool => Number::Bool(n.cast_into()),
            Self::Complex(ct) => Number::Complex(ct.cast(n)),
            Self::Float(ft) => Number::Float(ft.cast(n)),
            Self::Int(it) => Number::Int(it.cast(n)),
            Self::UInt(ut) => Number::UInt(ut.cast(n)),
            Self::Number => n,
        }
    }

    fn size(self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex(ct) => NumberClass::size(ct),
            Float(ft) => NumberClass::size(ft),
            Int(it) => NumberClass::size(it),
            UInt(ut) => NumberClass::size(ut),

            // a generic Number still has a distinct maximum size
            Number => NumberClass::size(ComplexType::C64),
        }
    }

    fn one(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => true.into(),
            Complex(ct) => ct.one().into(),
            Float(ft) => ft.one().into(),
            Int(it) => it.one().into(),
            UInt(ut) => ut.one().into(),
        }
    }

    fn zero(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => false.into(),
            Complex(ct) => ct.zero().into(),
            Float(ft) => ft.zero().into(),
            Int(it) => it.zero().into(),
            UInt(ut) => ut.zero().into(),
        }
    }
}

impl Ord for NumberType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Bool, Self::Bool) => Ordering::Equal,
            (Self::Complex(l), Self::Complex(r)) => l.cmp(r),
            (Self::Float(l), Self::Float(r)) => l.cmp(r),
            (Self::Int(l), Self::Int(r)) => l.cmp(r),
            (Self::UInt(l), Self::UInt(r)) => l.cmp(r),

            (Self::Number, Self::Number) => Ordering::Equal,
            (Self::Number, _) => Ordering::Greater,
            (_, Self::Number) => Ordering::Less,

            (Self::Bool, _) => Ordering::Less,
            (_, Self::Bool) => Ordering::Greater,

            (Self::Complex(_), _) => Ordering::Greater,
            (_, Self::Complex(_)) => Ordering::Less,

            (Self::UInt(_), Self::Int(_)) => Ordering::Less,
            (Self::UInt(_), Self::Float(_)) => Ordering::Less,
            (Self::Int(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Float(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Int(_), Self::Float(_)) => Ordering::Less,
            (Self::Float(_), Self::Int(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for NumberType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => fmt::Debug::fmt(&BooleanType, f),
            Complex(ct) => fmt::Debug::fmt(ct, f),
            Float(ft) => fmt::Debug::fmt(ft, f),
            Int(it) => fmt::Debug::fmt(it, f),
            UInt(ut) => fmt::Debug::fmt(ut, f),
            Number => write!(f, "Number"),
        }
    }
}
