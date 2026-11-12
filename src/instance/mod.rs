use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use std::str::FromStr;

use collate::Collate;
use get_size::GetSize;
use safecast::{CastFrom, CastInto};

use super::class::{
    BooleanType, ComplexType, FloatInstance, FloatType, IntType, NumberClass, NumberInstance,
    NumberType, RealInstance, Trigonometry, UIntType,
};
use super::{_Complex, Error, ErrorKind, Number};
use crate::checked;

macro_rules! fmt_debug {
    ($t:ty) => {
        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{} ({})", self, self.class())
            }
        }
    };
}

macro_rules! ordinary_methods {
    ($module:ident; $($name:ident => $out:ident),*) => {$(
        fn $name(self) -> Self::$out {
            crate::native::$module::$name(self)
        }
    )*};
}

macro_rules! primitive_binary {
    ($left:expr, $right:expr, $op:tt, $family:ident; $($variant:ident),+) => {{
        let dtype = Ord::max($left.class(), $right.class());
        match ($left.into_type(dtype), $right.into_type(dtype)) {
            $(($family::$variant(l), $family::$variant(r)) => $family::$variant(l $op r),)+
            _ => unreachable!("promoted operand widths"),
        }
    }};
}

macro_rules! operators {
    ($ty:ident; $($trait:ident, $method:ident, $assign:ident, $assign_method:ident, |$l:ident, $r:ident| $body:expr;)*) => {$(
        impl $trait for $ty {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                let $l = self;
                let $r = other;
                $body
            }
        }

        impl $assign for $ty {
            fn $assign_method(&mut self, other: Self) {
                let result = $trait::$method(*self, other);
                *self = result;
            }
        }
    )*};
}

macro_rules! ordinary_aggregates {
    ($($ty:ident),*) => {$(
        impl Sum for $ty {
            fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
                let Some(first) = iter.next() else {
                    return Self::default().class().zero();
                };
                iter.fold(first, |a, b| a + b)
            }
        }

        impl Product for $ty {
            fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
                let Some(first) = iter.next() else {
                    return Self::default().class().one();
                };
                iter.fold(first, |a, b| a * b)
            }
        }
    )*};
}

operators! { Boolean;
    Add, add, AddAssign, add_assign, |left, right| left.try_add(right).expect("Boolean arithmetic");
    Sub, sub, SubAssign, sub_assign, |left, right| left.try_sub(right).expect("Boolean arithmetic");
    Mul, mul, MulAssign, mul_assign, |left, right| left.try_mul(right).expect("Boolean arithmetic");
    Div, div, DivAssign, div_assign, |left, right| left.try_div(right).expect("Boolean arithmetic");
    Rem, rem, RemAssign, rem_assign, |left, right| left.try_rem(right).expect("Boolean arithmetic");
}

operators! { Complex;
    Add, add, AddAssign, add_assign, |left, right| primitive_binary!(left, right, +, Complex; C32, C64);
    Sub, sub, SubAssign, sub_assign, |left, right| primitive_binary!(left, right, -, Complex; C32, C64);
    Mul, mul, MulAssign, mul_assign, |left, right| primitive_binary!(left, right, *, Complex; C32, C64);
    Div, div, DivAssign, div_assign, |left, right| primitive_binary!(left, right, /, Complex; C32, C64);
    Rem, rem, RemAssign, rem_assign, |left, right| primitive_binary!(left, right, %, Complex; C32, C64);
}

operators! { Float;
    Add, add, AddAssign, add_assign, |left, right| primitive_binary!(left, right, +, Float; F32, F64);
    Sub, sub, SubAssign, sub_assign, |left, right| primitive_binary!(left, right, -, Float; F32, F64);
    Mul, mul, MulAssign, mul_assign, |left, right| primitive_binary!(left, right, *, Float; F32, F64);
    Div, div, DivAssign, div_assign, |left, right| primitive_binary!(left, right, /, Float; F32, F64);
    Rem, rem, RemAssign, rem_assign, |left, right| primitive_binary!(left, right, %, Float; F32, F64);
}

operators! { Int;
    Add, add, AddAssign, add_assign, |left, right| primitive_binary!(left, right, +, Int; I8, I16, I32, I64);
    Sub, sub, SubAssign, sub_assign, |left, right| primitive_binary!(left, right, -, Int; I8, I16, I32, I64);
    Mul, mul, MulAssign, mul_assign, |left, right| primitive_binary!(left, right, *, Int; I8, I16, I32, I64);
    Div, div, DivAssign, div_assign, |left, right| primitive_binary!(left, right, /, Int; I8, I16, I32, I64);
    Rem, rem, RemAssign, rem_assign, |left, right| primitive_binary!(left, right, %, Int; I8, I16, I32, I64);
}

operators! { UInt;
    Add, add, AddAssign, add_assign, |left, right| primitive_binary!(left, right, +, UInt; U8, U16, U32, U64);
    Sub, sub, SubAssign, sub_assign, |left, right| primitive_binary!(left, right, -, UInt; U8, U16, U32, U64);
    Mul, mul, MulAssign, mul_assign, |left, right| primitive_binary!(left, right, *, UInt; U8, U16, U32, U64);
    Div, div, DivAssign, div_assign, |left, right| primitive_binary!(left, right, /, UInt; U8, U16, U32, U64);
    Rem, rem, RemAssign, rem_assign, |left, right| primitive_binary!(left, right, %, UInt; U8, U16, U32, U64);
}

ordinary_aggregates!(Complex, Float, Int, UInt);

/// A boolean value.
#[derive(Clone, Copy, Default, Hash, Ord, PartialEq, PartialOrd)]
pub struct Boolean(bool);

impl GetSize for Boolean {
    fn get_size(&self) -> usize {
        1
    }
}

impl NumberInstance for Boolean {
    fn abs(self) -> Self::Abs {
        self
    }

    fn exp(self) -> Self::Exp {
        Float::from(self).exp()
    }

    fn ln(self) -> Self::Log {
        Float::from(self).ln()
    }

    fn round(self) -> Self::Round {
        self
    }

    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        crate::native::float::log(self.into(), base.into())
    }

    fn pow(self, exponent: Number) -> Self {
        self.try_pow(exponent).expect("Boolean power")
    }

    type Abs = Self;
    type Exp = Float;
    type Log = Float;
    type Round = Self;
    type Class = BooleanType;

    fn class(&self) -> BooleanType {
        BooleanType
    }

    fn into_type(self, _dtype: BooleanType) -> Boolean {
        self
    }

    fn and(self, other: Self) -> Self {
        Self(self.0 && other.0)
    }

    fn not(self) -> Self {
        Self(!self.0)
    }

    fn or(self, other: Self) -> Self {
        Self(self.0 || other.0)
    }

    fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }

    fn try_into_type(self, _dtype: BooleanType) -> Result<Self, Error> {
        let value = checked::into_type(self.into(), NumberType::Bool)?;
        let Number::Bool(value) = value else {
            unreachable!("resolved conversion family")
        };
        Ok(value)
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        Ok(self.or(other))
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        Ok(self.xor(other))
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        Ok(self.and(other))
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        Ok(self.xor(other))
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        if other.0 {
            Ok(self)
        } else {
            Err(checked::error(
                ErrorKind::DivisionByZero,
                "Boolean division",
            ))
        }
    }

    fn try_abs(self) -> Result<Self::Abs, Error> {
        Ok(self)
    }

    fn try_round(self) -> Result<Self::Round, Error> {
        Ok(self)
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        checked::pow_unsigned(UInt::from(self), exponent)
            .map(|value| Boolean::from(u64::from(value) != 0))
    }

    fn try_exp(self) -> Result<Self::Exp, Error> {
        Float::from(self).try_exp()
    }

    fn try_ln(self) -> Result<Self::Log, Error> {
        Float::from(self).try_ln()
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        checked::log(self.into(), base.into())
    }

    fn try_sum<I: Iterator<Item = Self>>(mut iter: I) -> Result<Self, Error> {
        Ok(Self(iter.any(|x| x.0)))
    }

    fn try_product<I: Iterator<Item = Self>>(mut iter: I) -> Result<Self, Error> {
        Ok(Self(iter.all(|x| x.0)))
    }
}

impl RealInstance for Boolean {
    const ONE: Self = Boolean(true);
    const ZERO: Self = Boolean(false);
}

impl From<bool> for Boolean {
    fn from(b: bool) -> Boolean {
        Self(b)
    }
}

impl FromStr for Boolean {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        bool::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl From<Boolean> for bool {
    fn from(b: Boolean) -> bool {
        b.0
    }
}

impl From<&Boolean> for bool {
    fn from(b: &Boolean) -> bool {
        b.0
    }
}

impl Eq for Boolean {}

impl Sum for Boolean {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::try_sum(iter).expect("numeric sum failed")
    }
}

impl Product for Boolean {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self::try_product(iter).expect("numeric product failed")
    }
}

impl CastFrom<Boolean> for u64 {
    fn cast_from(b: Boolean) -> u64 {
        UInt::from(b).into()
    }
}

fmt_debug!(Boolean);

impl fmt::Display for Boolean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// A complex number.
#[derive(Clone, Copy)]
pub enum Complex {
    C32(_Complex<f32>),
    C64(_Complex<f64>),
}

impl GetSize for Complex {
    fn get_size(&self) -> usize {
        match self {
            Self::C32(c) => c.re.get_size() + c.im.get_size(),
            Self::C64(c) => c.re.get_size() + c.im.get_size(),
        }
    }
}

impl Complex {
    /// Borrow the imaginary part of this [`Complex`] number
    pub fn im(&self) -> Float {
        match self {
            Self::C32(c) => Float::F32(c.im),
            Self::C64(c) => Float::F64(c.im),
        }
    }

    /// Borrow the real part of this [`Complex`] number
    pub fn re(&self) -> Float {
        match self {
            Self::C32(c) => Float::F32(c.re),
            Self::C64(c) => Float::F64(c.re),
        }
    }
}

impl NumberInstance for Complex {
    ordinary_methods!(complex; abs => Abs, exp => Exp, ln => Log, round => Round);

    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        crate::native::complex::log(self, base.into())
    }

    fn pow(self, exponent: Number) -> Self {
        crate::native::complex::pow(self, exponent)
    }

    type Abs = Float;
    type Exp = Self;
    type Log = Self;
    type Round = Self;
    type Class = ComplexType;

    fn class(&self) -> ComplexType {
        match self {
            Self::C32(_) => ComplexType::C32,
            Self::C64(_) => ComplexType::C64,
        }
    }

    fn into_type(self, dtype: ComplexType) -> Complex {
        use ComplexType::*;
        match dtype {
            C32 => match self {
                Self::C64(c) => Self::C32(_Complex::new(c.re as f32, c.im as f32)),
                this => this,
            },
            C64 => match self {
                Self::C32(c) => Self::C64(_Complex::new(c.re as f64, c.im as f64)),
                this => this,
            },
            Complex => self,
        }
    }

    fn try_into_type(self, dtype: ComplexType) -> Result<Self, Error> {
        let value = checked::into_type(self.into(), NumberType::Complex(dtype))?;
        let Number::Complex(value) = value else {
            unreachable!("resolved conversion family")
        };
        Ok(value)
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        if !checked::complex_finite(left) || !checked::complex_finite(right) {
            return Ok(match (left, right) {
                (Self::C32(l), Self::C32(r)) => Self::C32(l + r),
                (Self::C64(l), Self::C64(r)) => Self::C64(l + r),
                _ => unreachable!("complex width"),
            });
        }
        let re = left.re().try_add(right.re())?;
        let im = left.im().try_add(right.im())?;
        Ok(match (re, im) {
            (Float::F32(re), Float::F32(im)) => Self::C32(_Complex::new(re, im)),
            (Float::F64(re), Float::F64(im)) => Self::C64(_Complex::new(re, im)),
            _ => unreachable!("complex width"),
        })
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        if !checked::complex_finite(left) || !checked::complex_finite(right) {
            return Ok(match (left, right) {
                (Self::C32(l), Self::C32(r)) => Self::C32(l - r),
                (Self::C64(l), Self::C64(r)) => Self::C64(l - r),
                _ => unreachable!("complex width"),
            });
        }
        let re = left.re().try_sub(right.re())?;
        let im = left.im().try_sub(right.im())?;
        Ok(match (re, im) {
            (Float::F32(re), Float::F32(im)) => Self::C32(_Complex::new(re, im)),
            (Float::F64(re), Float::F64(im)) => Self::C64(_Complex::new(re, im)),
            _ => unreachable!("complex width"),
        })
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        if checked::complex_finite(left)
            && checked::complex_finite(right)
            && f64::from(left.im()) == 0.0
            && f64::from(right.im()) == 0.0
        {
            let real = left.re().try_mul(right.re())?;
            let im = left.im() * right.re() + left.re() * right.im();
            return Ok(crate::native::parts(real, im));
        }
        let result = match (left, right) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l * r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l * r),
            _ => unreachable!("complex width"),
        };
        checked::complex_binary(
            left,
            right,
            result,
            |a, b, c, d| (&a * &c - &b * &d, a * d + b * c),
            "mul",
        )
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        if checked::complex_finite(left)
            && checked::complex_finite(right)
            && f64::from(right.re()) == 0.0
            && f64::from(right.im()) == 0.0
        {
            return Err(checked::error(ErrorKind::DivisionByZero, "div"));
        }
        if checked::complex_finite(left)
            && checked::complex_finite(right)
            && f64::from(left.im()) == 0.0
            && f64::from(right.im()) == 0.0
        {
            let real = left.re().try_div(right.re())?;
            let im = left.im() * right.re() - left.re() * right.im();
            return Ok(crate::native::parts(real, im));
        }
        let result = match (left, right) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l / r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l / r),
            _ => unreachable!("complex width"),
        };
        checked::complex_binary(
            left,
            right,
            result,
            |a, b, c, d| {
                let denominator = &c * &c + &d * &d;
                (
                    (&a * &c + &b * &d) / &denominator,
                    (b * c - a * d) / denominator,
                )
            },
            "div",
        )
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        if checked::complex_finite(left)
            && checked::complex_finite(right)
            && f64::from(right.re()) == 0.0
            && f64::from(right.im()) == 0.0
        {
            return Err(checked::error(ErrorKind::DivisionByZero, "rem"));
        }
        let result = match (left, right) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l % r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l % r),
            _ => unreachable!("complex width"),
        };
        checked::complex_binary(
            left,
            right,
            result,
            |a, b, c, d| {
                let denominator = &c * &c + &d * &d;
                let re = (&a * &c + &b * &d) / &denominator;
                let im = (&b * &c - &a * &d) / denominator;
                let re = num::BigRational::from_integer(re.to_integer());
                let im = num::BigRational::from_integer(im.to_integer());
                (a - (&c * &re - &d * &im), b - (c * im + d * re))
            },
            "rem",
        )
    }

    fn try_abs(self) -> Result<Self::Abs, Error> {
        let result = checked::math::hypot(self.re(), self.im());
        checked::float_result(
            checked::complex_finite(self),
            result,
            f64::from(self.re()) == 0.0 && f64::from(self.im()) == 0.0,
            ErrorKind::Underflow,
            "abs",
        )
    }

    fn try_exp(self) -> Result<Self::Exp, Error> {
        if let Some(result) = checked::math::exp_axis(self, Float::try_exp)? {
            return Ok(result);
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "checked complex exp requires a finite real axis",
        ))
    }

    fn try_ln(self) -> Result<Self::Log, Error> {
        if let Some(result) = checked::math::ln_axis(self, Float::try_ln)? {
            return Ok(result);
        }
        if !checked::complex_finite(self) {
            return Err(checked::error(
                ErrorKind::Unsupported,
                "nonfinite complex ln",
            ));
        }
        let (re, im) = (self.re(), self.im());
        let magnitude = if f64::from(im) == 0.0 {
            re.try_abs()?
        } else if f64::from(re) == 0.0 {
            im.try_abs()?
        } else {
            return Err(checked::error(
                ErrorKind::Unsupported,
                "checked complex ln requires a coordinate axis",
            ));
        };
        Ok(crate::native::parts(
            magnitude.try_ln()?,
            checked::math::atan2(im, re),
        ))
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        let base = base.into();
        if matches!(base, Number::Complex(_)) {
            return Err(checked::error(
                ErrorKind::Unsupported,
                "checked complex logarithm base",
            ));
        }
        let dtype = Ord::max(checked::float_type(self.into()), checked::float_type(base));
        let this = self.into_type(if dtype == FloatType::F64 {
            ComplexType::C64
        } else {
            ComplexType::C32
        });
        let Number::Float(base) = checked::cast(base, dtype.into())? else {
            unreachable!("prepared logarithm base")
        };
        if !checked::complex_finite(this) || !checked::finite(base) {
            return Err(checked::error(
                ErrorKind::Unsupported,
                "nonfinite complex log",
            ));
        }
        if f64::from(base) <= 0.0 || f64::from(base) == 1.0 {
            return Err(checked::error(ErrorKind::Domain, "log"));
        }
        let value = this.try_ln()?;
        let denominator = base.try_ln()?;
        Ok(crate::native::parts(
            value.re().try_div(denominator)?,
            value.im().try_div(denominator)?,
        ))
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        if matches!(exponent, Number::Complex(_)) {
            return Err(checked::error(ErrorKind::Unsupported, "complex exponent"));
        }
        let finite_exponent = match exponent {
            Number::Float(value) => checked::finite(value),
            _ => true,
        };
        if checked::complex_finite(self)
            && f64::from(self.re()) == 0.0
            && f64::from(self.im()) == 0.0
            && finite_exponent
            && exponent < Number::from(0)
        {
            return Err(checked::error(ErrorKind::DivisionByZero, "pow"));
        }
        let integral = crate::native::integer_exponent(exponent).map_err(|_| {
            checked::error(
                ErrorKind::Unsupported,
                "checked complex powers require a bounded integral exponent",
            )
        })?;
        let this = if matches!(exponent, Number::Float(Float::F64(_))) {
            self.into_type(ComplexType::C64)
        } else {
            self
        };
        let one = this.class().one();
        let base = if integral.negative {
            one.try_div(this)?
        } else {
            this
        };
        crate::native::power(base, integral.magnitude, one, Self::try_mul)
    }

    fn try_round(self) -> Result<Self::Round, Error> {
        Ok(crate::native::parts(
            self.re().try_round()?,
            self.im().try_round()?,
        ))
    }
}

impl FloatInstance for Complex {
    fn is_infinite(&self) -> bool {
        match self {
            Self::C32(c) => c.im.is_infinite() || c.re.is_infinite(),
            Self::C64(c) => c.im.is_infinite() || c.re.is_infinite(),
        }
    }

    fn is_nan(&self) -> bool {
        match self {
            Self::C32(c) => c.im.is_nan() || c.re.is_nan(),
            Self::C64(c) => c.im.is_nan() || c.re.is_nan(),
        }
    }
}

impl From<[f32; 2]> for Complex {
    fn from(arr: [f32; 2]) -> Self {
        Self::C32(num::Complex::new(arr[0], arr[1]))
    }
}

impl From<[f64; 2]> for Complex {
    fn from(arr: [f64; 2]) -> Self {
        Self::C64(num::Complex::new(arr[0], arr[1]))
    }
}

impl<R, I> From<(R, I)> for Complex
where
    R: Into<f64>,
    I: Into<f64>,
{
    fn from(value: (R, I)) -> Self {
        let re = value.0.into();
        let im = value.1.into();
        Self::C64(num::Complex::new(re, im))
    }
}

impl CastFrom<Number> for Complex {
    fn cast_from(number: Number) -> Complex {
        use Number::*;
        match number {
            Number::Bool(b) => Self::from(b),
            Complex(c) => c,
            Float(f) => Self::from(f),
            Int(i) => Self::from(i),
            UInt(u) => Self::from(u),
        }
    }
}

impl CastFrom<Complex> for Boolean {
    fn cast_from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => Self(c.re != 0.0 || c.im != 0.0),
            Complex::C64(c) => Self(c.re != 0.0 || c.im != 0.0),
        }
    }
}

impl CastFrom<Complex> for _Complex<f32> {
    fn cast_from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => c,
            Complex::C64(_Complex { re, im }) => Self::new(re as f32, im as f32),
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        let l: _Complex<f64> = (*self).cast_into();
        let r: _Complex<f64> = (*other).cast_into();

        super::normalized_f64_bits(l.re) == super::normalized_f64_bits(r.re)
            && super::normalized_f64_bits(l.im) == super::normalized_f64_bits(r.im)
    }
}

impl Eq for Complex {}

impl Hash for Complex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let c: _Complex<f64> = (*self).cast_into();
        super::normalized_f64_bits(c.re).hash(state);
        super::normalized_f64_bits(c.im).hash(state);
    }
}

impl Default for Complex {
    fn default() -> Complex {
        Complex::C32(_Complex::<f32>::default())
    }
}

impl From<Complex> for _Complex<f64> {
    fn from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => Self::new(c.re as f64, c.im as f64),
            Complex::C64(c64) => c64,
        }
    }
}

impl From<Float> for Complex {
    fn from(f: Float) -> Self {
        match f {
            Float::F64(f) => Self::C64(_Complex::new(f, 0.0f64)),
            Float::F32(f) => Self::C32(_Complex::new(f, 0.0f32)),
        }
    }
}

impl From<Int> for Complex {
    fn from(i: Int) -> Self {
        match i {
            Int::I8(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
            Int::I64(i) => Self::C64(_Complex::new(i as f64, 0.0f64)),
            Int::I32(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
            Int::I16(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
        }
    }
}

impl From<UInt> for Complex {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::C64(_Complex::new(u as f64, 0.0f64)),
            UInt::U32(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
            UInt::U16(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
            UInt::U8(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
        }
    }
}

impl From<Boolean> for Complex {
    fn from(b: Boolean) -> Self {
        match b {
            Boolean(true) => Self::C32(_Complex::new(1.0f32, 0.0f32)),
            Boolean(false) => Self::C32(_Complex::new(0.0f32, 0.0f32)),
        }
    }
}

impl From<_Complex<f32>> for Complex {
    fn from(c: _Complex<f32>) -> Self {
        Self::C32(c)
    }
}

impl From<_Complex<f64>> for Complex {
    fn from(c: _Complex<f64>) -> Self {
        Self::C64(c)
    }
}

impl FromStr for Complex {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        num::Complex::<f64>::from_str(s)
            .map(Self::from)
            .map_err(Error::new)
    }
}

impl From<Complex> for (Float, Float) {
    fn from(n: Complex) -> Self {
        match n {
            Complex::C32(n) => (Float::F32(n.re), Float::F32(n.im)),
            Complex::C64(n) => (Float::F64(n.re), Float::F64(n.im)),
        }
    }
}

impl From<Complex> for [Float; 2] {
    fn from(n: Complex) -> Self {
        match n {
            Complex::C32(n) => [Float::F32(n.re), Float::F32(n.im)],
            Complex::C64(n) => [Float::F64(n.re), Float::F64(n.im)],
        }
    }
}

fmt_debug!(Complex);

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Complex::C32(c) => fmt::Display::fmt(c, f),
            Complex::C64(c) => fmt::Display::fmt(c, f),
        }
    }
}

/// Defines a collation order for [`Complex`].
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct ComplexCollator {
    float: FloatCollator,
}

impl Collate for ComplexCollator {
    type Value = Complex;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        let left: _Complex<f64> = (*left).into();
        let right: _Complex<f64> = (*right).into();
        self.float.cmp(
            &Float::F64(left.re.hypot(left.im)),
            &Float::F64(right.re.hypot(right.im)),
        )
    }
}

/// A floating-point number.
#[derive(Clone, Copy, get_size_derive::GetSize)]
pub enum Float {
    F32(f32),
    F64(f64),
}

impl NumberInstance for Float {
    ordinary_methods!(float; abs => Abs, exp => Exp, ln => Log, round => Round);

    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        crate::native::float::log(self.into(), base.into())
    }

    fn pow(self, exponent: Number) -> Self {
        crate::native::float::pow(self, exponent)
    }

    type Abs = Float;
    type Exp = Self;
    type Log = Self;
    type Round = Self;
    type Class = FloatType;

    fn class(&self) -> FloatType {
        match self {
            Self::F32(_) => FloatType::F32,
            Self::F64(_) => FloatType::F64,
        }
    }

    fn into_type(self, dtype: FloatType) -> Float {
        use FloatType::*;
        match dtype {
            F32 => match self {
                Self::F64(f) => Self::F32(f as f32),
                this => this,
            },
            F64 => match self {
                Self::F32(f) => Self::F64(f as f64),
                this => this,
            },
            Float => self,
        }
    }

    fn try_into_type(self, dtype: FloatType) -> Result<Self, Error> {
        let value = checked::into_type(self.into(), NumberType::Float(dtype))?;
        let Number::Float(value) = value else {
            unreachable!("resolved conversion family")
        };
        Ok(value)
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        let finite = checked::finite(left) && checked::finite(right);
        let result = match (left, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left + right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left + right),
            _ => unreachable!("promoted floating width"),
        };
        checked::float_result(
            finite,
            result,
            f64::from(left) == -f64::from(right),
            ErrorKind::Underflow,
            "add",
        )
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        let finite = checked::finite(left) && checked::finite(right);
        let result = match (left, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left - right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left - right),
            _ => unreachable!("promoted floating width"),
        };
        checked::float_result(finite, result, left == right, ErrorKind::Underflow, "sub")
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        let finite = checked::finite(left) && checked::finite(right);
        let result = match (left, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left * right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left * right),
            _ => unreachable!("promoted floating width"),
        };
        checked::float_result(
            finite,
            result,
            f64::from(left) == 0.0 || f64::from(right) == 0.0,
            ErrorKind::Underflow,
            "mul",
        )
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        let finite = checked::finite(left) && checked::finite(right);
        if finite && f64::from(right) == 0.0 {
            return Err(checked::error(ErrorKind::DivisionByZero, "div"));
        }
        let result = match (left, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left / right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left / right),
            _ => unreachable!("promoted floating width"),
        };
        checked::float_result(
            finite,
            result,
            f64::from(left) == 0.0,
            ErrorKind::Underflow,
            "div",
        )
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        let dtype = Ord::max(self.class(), other.class());
        let left = self.into_type(dtype);
        let right = other.into_type(dtype);
        let finite = checked::finite(left) && checked::finite(right);
        if finite && f64::from(right) == 0.0 {
            return Err(checked::error(ErrorKind::DivisionByZero, "rem"));
        }
        let result = match (left, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left % right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left % right),
            _ => unreachable!("promoted floating width"),
        };
        checked::float_result(finite, result, true, ErrorKind::Underflow, "rem")
    }

    fn try_abs(self) -> Result<Self::Abs, Error> {
        Ok(checked::math::abs(self))
    }

    fn try_exp(self) -> Result<Self::Exp, Error> {
        let result = checked::math::exp(self);
        checked::float_result(
            checked::finite(self),
            result,
            false,
            ErrorKind::Underflow,
            "exp",
        )
    }

    fn try_ln(self) -> Result<Self::Log, Error> {
        let value = f64::from(self);
        if checked::finite(self) && value <= 0.0 {
            return Err(checked::error(ErrorKind::Domain, "ln"));
        }
        let result = checked::math::ln(self);
        checked::float_result(
            checked::finite(self),
            result,
            value == 1.0,
            ErrorKind::Underflow,
            "ln",
        )
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        checked::log(self.into(), base.into())
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        if matches!(exponent, Number::Complex(_)) {
            return Err(checked::error(ErrorKind::Unsupported, "complex exponent"));
        }
        if !matches!(exponent, Number::Float(_)) {
            let exponent = crate::native::integer_exponent(exponent)?;
            let one = self.class().one();
            let base = if exponent.negative {
                one.try_div(self)?
            } else {
                self
            };
            return crate::native::power(base, exponent.magnitude, one, Self::try_mul);
        }
        let dtype = if matches!(exponent, Number::Float(Self::F64(_))) {
            FloatType::F64
        } else {
            self.class()
        };
        let left = self.into_type(dtype);
        let Number::Float(right) = checked::cast(exponent, NumberType::Float(dtype))? else {
            unreachable!("float exponent")
        };
        let finite = checked::finite(left) && checked::finite(right);
        let x = f64::from(left);
        let y = f64::from(right);
        if finite && x == 0.0 && y < 0.0 {
            return Err(checked::error(ErrorKind::DivisionByZero, "pow"));
        }
        if finite && x < 0.0 && y.fract() != 0.0 {
            return Err(checked::error(ErrorKind::Domain, "pow"));
        }
        let result = checked::math::pow(left, right);
        checked::float_result(
            finite,
            result,
            x == 0.0 && y > 0.0,
            ErrorKind::Underflow,
            "pow",
        )
    }

    fn try_round(self) -> Result<Self::Round, Error> {
        Ok(checked::math::round(self))
    }
}

impl RealInstance for Float {
    const ONE: Self = Float::F32(1.);
    const ZERO: Self = Float::F32(0.);
}

impl FloatInstance for Float {
    fn is_infinite(&self) -> bool {
        match self {
            Self::F32(f) => f.is_infinite(),
            Self::F64(f) => f.is_infinite(),
        }
    }

    fn is_nan(&self) -> bool {
        match self {
            Self::F32(f) => f.is_nan(),
            Self::F64(f) => f.is_nan(),
        }
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        super::normalized_f64_bits(f64::from(*self)).hash(state);
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        super::normalized_f64_bits(f64::from(*self))
            == super::normalized_f64_bits(f64::from(*other))
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (Float::F32(l), Float::F32(r)) => l.partial_cmp(r),
            (Float::F64(l), Float::F64(r)) => l.partial_cmp(r),
            (l, r) => f64::from(*l).partial_cmp(&f64::from(*r)),
        }
    }
}

impl Default for Float {
    fn default() -> Float {
        Float::F32(f32::default())
    }
}

impl From<Boolean> for Float {
    fn from(b: Boolean) -> Self {
        match b {
            Boolean(true) => Self::F32(1.0f32),
            Boolean(false) => Self::F32(0.0f32),
        }
    }
}

impl From<f32> for Float {
    fn from(f: f32) -> Self {
        Self::F32(f)
    }
}

impl From<f64> for Float {
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

impl From<Int> for Float {
    fn from(i: Int) -> Self {
        match i {
            Int::I8(i) => Self::F32(i as f32),
            Int::I64(i) => Self::F64(i as f64),
            Int::I32(i) => Self::F32(i as f32),
            Int::I16(i) => Self::F32(i as f32),
        }
    }
}

impl From<UInt> for Float {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::F64(u as f64),
            UInt::U32(u) => Self::F32(u as f32),
            UInt::U16(u) => Self::F32(u as f32),
            UInt::U8(u) => Self::F32(u as f32),
        }
    }
}

impl FromStr for Float {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl CastFrom<Complex> for Float {
    fn cast_from(c: Complex) -> Float {
        use Complex::*;
        match c {
            C32(c) => Self::F32(c.re),
            C64(c) => Self::F64(c.re),
        }
    }
}

impl CastFrom<Float> for Boolean {
    fn cast_from(f: Float) -> Boolean {
        use Float::*;
        Boolean(!matches!(f, F32(0f32) | F64(0f64)))
    }
}

impl CastFrom<Float> for f32 {
    fn cast_from(f: Float) -> f32 {
        match f {
            Float::F32(f) => f,
            Float::F64(f) => f as f32,
        }
    }
}

impl From<Float> for f64 {
    fn from(f: Float) -> f64 {
        match f {
            Float::F32(f) => f as f64,
            Float::F64(f) => f,
        }
    }
}

fmt_debug!(Float);

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Float::F32(n) => fmt::Display::fmt(n, f),
            Float::F64(n) => fmt::Display::fmt(n, f),
        }
    }
}

/// Defines a collation order for [`Float`]. NaNs compare equally, after other values.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct FloatCollator;

impl Collate for FloatCollator {
    type Value = Float;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        // Float comparison already equates NaNs; an unordered pair has exactly one.
        if let Some(order) = left.partial_cmp(right) {
            order
        } else if left.is_nan() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

/// A signed integer.
#[derive(Clone, Copy, get_size_derive::GetSize)]
pub enum Int {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

impl NumberInstance for Int {
    fn abs(self) -> Self::Abs {
        match self {
            Self::I8(x) => Self::I8(x.abs()),
            Self::I16(x) => Self::I16(x.abs()),
            Self::I32(x) => Self::I32(x.abs()),
            Self::I64(x) => Self::I64(x.abs()),
        }
    }

    fn exp(self) -> Self::Exp {
        Float::from(self).exp()
    }

    fn ln(self) -> Self::Log {
        Float::from(self).ln()
    }

    fn round(self) -> Self::Round {
        self
    }

    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        crate::native::float::log(self.into(), base.into())
    }

    fn pow(self, exponent: Number) -> Self {
        crate::native::int_pow(self, exponent)
    }

    type Abs = Self;
    type Exp = Float;
    type Log = Float;
    type Round = Self;
    type Class = IntType;

    fn class(&self) -> IntType {
        match self {
            Self::I8(_) => IntType::I8,
            Self::I16(_) => IntType::I16,
            Self::I32(_) => IntType::I32,
            Self::I64(_) => IntType::I64,
        }
    }

    fn into_type(self, dtype: IntType) -> Int {
        use IntType::*;
        match dtype {
            I8 => match self {
                Self::I16(i) => Self::I8(i as i8),
                Self::I32(i) => Self::I8(i as i8),
                Self::I64(i) => Self::I8(i as i8),
                this => this,
            },
            I16 => match self {
                Self::I8(i) => Self::I16(i as i16),
                Self::I32(i) => Self::I16(i as i16),
                Self::I64(i) => Self::I16(i as i16),
                this => this,
            },
            I32 => match self {
                Self::I8(i) => Self::I32(i as i32),
                Self::I16(i) => Self::I32(i as i32),
                Self::I64(i) => Self::I32(i as i32),
                this => this,
            },
            I64 => match self {
                Self::I8(i) => Self::I64(i as i64),
                Self::I16(i) => Self::I64(i as i64),
                Self::I32(i) => Self::I64(i as i64),
                this => this,
            },
            Int => self,
        }
    }

    fn try_into_type(self, dtype: IntType) -> Result<Self, Error> {
        let value = checked::into_type(self.into(), NumberType::Int(dtype))?;
        let Number::Int(value) = value else {
            unreachable!("resolved conversion family")
        };
        Ok(value)
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        checked::add_signed(
            i64::from(self) as i128,
            i64::from(other) as i128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        checked::sub_signed(
            i64::from(self) as i128,
            i64::from(other) as i128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        checked::mul_signed(
            i64::from(self) as i128,
            i64::from(other) as i128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        checked::div_signed(
            i64::from(self) as i128,
            i64::from(other) as i128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        checked::rem_signed(
            i64::from(self) as i128,
            i64::from(other) as i128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_abs(self) -> Result<Self::Abs, Error> {
        checked::signed(i128::from(i64::from(self)).abs(), self.class())
    }

    fn try_round(self) -> Result<Self::Round, Error> {
        Ok(self)
    }

    fn try_exp(self) -> Result<Self::Exp, Error> {
        Float::from(self).try_exp()
    }

    fn try_ln(self) -> Result<Self::Log, Error> {
        Float::from(self).try_ln()
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        checked::log(self.into(), base.into())
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        checked::pow_signed(self, exponent)
    }
}

impl RealInstance for Int {
    const ONE: Self = Int::I8(1);
    const ZERO: Self = Int::I8(0);
}

impl CastFrom<Complex> for Int {
    fn cast_from(c: Complex) -> Int {
        use Complex::*;
        match c {
            C32(c) => Self::I32(c.re as i32),
            C64(c) => Self::I64(c.re as i64),
        }
    }
}

impl CastFrom<Float> for Int {
    fn cast_from(f: Float) -> Int {
        use Float::*;
        match f {
            F32(f) => Self::I32(f as i32),
            F64(f) => Self::I64(f as i64),
        }
    }
}

impl CastFrom<Int> for Boolean {
    fn cast_from(i: Int) -> Boolean {
        use Int::*;
        Boolean(!matches!(i, I8(0) | I16(0) | I32(0) | I64(0)))
    }
}

impl CastFrom<Int> for i8 {
    fn cast_from(i: Int) -> i8 {
        match i {
            Int::I8(i) => i,
            Int::I16(i) => i as i8,
            Int::I32(i) => i as i8,
            Int::I64(i) => i as i8,
        }
    }
}

impl CastFrom<Int> for i16 {
    fn cast_from(i: Int) -> i16 {
        match i {
            Int::I8(i) => i as i16,
            Int::I16(i) => i,
            Int::I32(i) => i as i16,
            Int::I64(i) => i as i16,
        }
    }
}

impl CastFrom<Int> for i32 {
    fn cast_from(i: Int) -> i32 {
        match i {
            Int::I8(i) => i as i32,
            Int::I16(i) => i as i32,
            Int::I32(i) => i,
            Int::I64(i) => i as i32,
        }
    }
}

impl Eq for Int {}

impl PartialEq for Int {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I16(l), Self::I16(r)) => l.eq(r),
            (Self::I32(l), Self::I32(r)) => l.eq(r),
            (Self::I64(l), Self::I64(r)) => l.eq(r),
            (Self::I64(l), r) => l.eq(&i64::from(*r)),
            (l, r) => i64::from(*l).eq(&i64::from(*r)),
        }
    }
}

impl Hash for Int {
    fn hash<H: Hasher>(&self, state: &mut H) {
        i64::from(*self).hash(state);
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Int::I8(l), Int::I8(r)) => l.cmp(r),
            (Int::I16(l), Int::I16(r)) => l.cmp(r),
            (Int::I32(l), Int::I32(r)) => l.cmp(r),
            (Int::I64(l), Int::I64(r)) => l.cmp(r),
            (l, r) => i64::from(*l).cmp(&i64::from(*r)),
        }
    }
}

impl Default for Int {
    fn default() -> Int {
        Int::I16(i16::default())
    }
}

impl From<i8> for Int {
    fn from(i: i8) -> Int {
        Int::I8(i)
    }
}

impl From<i16> for Int {
    fn from(i: i16) -> Int {
        Int::I16(i)
    }
}

impl From<i32> for Int {
    fn from(i: i32) -> Int {
        Int::I32(i)
    }
}

impl From<i64> for Int {
    fn from(i: i64) -> Int {
        Int::I64(i)
    }
}

impl CastFrom<UInt> for Int {
    fn cast_from(u: UInt) -> Int {
        match u {
            UInt::U64(u) => Int::I64(u as i64),
            UInt::U32(u) => Int::I32(u as i32),
            UInt::U16(u) => Int::I16(u as i16),
            UInt::U8(u) => Int::I16(u as i16),
        }
    }
}

impl From<Boolean> for Int {
    fn from(b: Boolean) -> Int {
        match b {
            Boolean(true) => Int::I16(1),
            Boolean(false) => Int::I16(0),
        }
    }
}

impl From<Int> for i64 {
    fn from(i: Int) -> i64 {
        match i {
            Int::I8(i) => i as i64,
            Int::I16(i) => i as i64,
            Int::I32(i) => i as i64,
            Int::I64(i) => i,
        }
    }
}

impl FromStr for Int {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        i64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

fmt_debug!(Int);

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Int::I8(i) => fmt::Display::fmt(i, f),
            Int::I16(i) => fmt::Display::fmt(i, f),
            Int::I32(i) => fmt::Display::fmt(i, f),
            Int::I64(i) => fmt::Display::fmt(i, f),
        }
    }
}

/// An unsigned integer.
#[derive(Clone, Copy, get_size_derive::GetSize)]
pub enum UInt {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl NumberInstance for UInt {
    fn abs(self) -> Self::Abs {
        self
    }

    fn exp(self) -> Self::Exp {
        Float::from(self).exp()
    }

    fn ln(self) -> Self::Log {
        Float::from(self).ln()
    }

    fn round(self) -> Self::Round {
        self
    }

    fn log<N: NumberInstance>(self, base: N) -> Self::Log {
        crate::native::float::log(self.into(), base.into())
    }

    fn pow(self, exponent: Number) -> Self {
        crate::native::uint_pow(self, exponent)
    }

    type Abs = Self;
    type Exp = Float;
    type Log = Float;
    type Round = Self;
    type Class = UIntType;

    fn class(&self) -> UIntType {
        match self {
            Self::U8(_) => UIntType::U8,
            Self::U16(_) => UIntType::U16,
            Self::U32(_) => UIntType::U32,
            Self::U64(_) => UIntType::U64,
        }
    }

    fn into_type(self, dtype: UIntType) -> UInt {
        use UIntType::*;
        match dtype {
            U8 => match self {
                Self::U16(u) => Self::U8(u as u8),
                Self::U32(u) => Self::U8(u as u8),
                Self::U64(u) => Self::U8(u as u8),
                this => this,
            },
            U16 => match self {
                Self::U8(u) => Self::U16(u as u16),
                Self::U32(u) => Self::U16(u as u16),
                Self::U64(u) => Self::U16(u as u16),
                this => this,
            },
            U32 => match self {
                Self::U8(u) => Self::U32(u as u32),
                Self::U16(u) => Self::U32(u as u32),
                Self::U64(u) => Self::U32(u as u32),
                this => this,
            },
            U64 => match self {
                Self::U8(u) => Self::U64(u as u64),
                Self::U16(u) => Self::U64(u as u64),
                Self::U32(u) => Self::U64(u as u64),
                this => this,
            },
            UInt => self,
        }
    }

    fn try_into_type(self, dtype: UIntType) -> Result<Self, Error> {
        let value = checked::into_type(self.into(), NumberType::UInt(dtype))?;
        let Number::UInt(value) = value else {
            unreachable!("resolved conversion family")
        };
        Ok(value)
    }

    fn try_add(self, other: Self) -> Result<Self, Error> {
        checked::add_unsigned(
            u64::from(self) as u128,
            u64::from(other) as u128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_sub(self, other: Self) -> Result<Self, Error> {
        checked::sub_unsigned(
            u64::from(self) as u128,
            u64::from(other) as u128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_mul(self, other: Self) -> Result<Self, Error> {
        checked::mul_unsigned(
            u64::from(self) as u128,
            u64::from(other) as u128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_div(self, other: Self) -> Result<Self, Error> {
        checked::div_unsigned(
            u64::from(self) as u128,
            u64::from(other) as u128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_rem(self, other: Self) -> Result<Self, Error> {
        checked::rem_unsigned(
            u64::from(self) as u128,
            u64::from(other) as u128,
            Ord::max(self.class(), other.class()),
        )
    }

    fn try_abs(self) -> Result<Self::Abs, Error> {
        Ok(self)
    }

    fn try_round(self) -> Result<Self::Round, Error> {
        Ok(self)
    }

    fn try_exp(self) -> Result<Self::Exp, Error> {
        Float::from(self).try_exp()
    }

    fn try_ln(self) -> Result<Self::Log, Error> {
        Float::from(self).try_ln()
    }

    fn try_log<N: NumberInstance>(self, base: N) -> Result<Self::Log, Error> {
        checked::log(self.into(), base.into())
    }

    fn try_pow(self, exponent: Number) -> Result<Self, Error> {
        checked::pow_unsigned(self, exponent)
    }
}

impl RealInstance for UInt {
    const ONE: Self = UInt::U8(1);
    const ZERO: Self = UInt::U8(0);
}

impl CastFrom<Complex> for UInt {
    fn cast_from(c: Complex) -> UInt {
        use Complex::*;
        match c {
            C32(c) => Self::U32(c.re as u32),
            C64(c) => Self::U64(c.re as u64),
        }
    }
}

impl CastFrom<Float> for UInt {
    fn cast_from(f: Float) -> UInt {
        use Float::*;
        match f {
            F32(f) => Self::U32(f as u32),
            F64(f) => Self::U64(f as u64),
        }
    }
}

impl CastFrom<Int> for UInt {
    fn cast_from(i: Int) -> UInt {
        use Int::*;
        match i {
            I8(i) => Self::U8(i as u8),
            I16(i) => Self::U16(i as u16),
            I32(i) => Self::U32(i as u32),
            I64(i) => Self::U64(i as u64),
        }
    }
}

impl CastFrom<UInt> for bool {
    fn cast_from(u: UInt) -> bool {
        use UInt::*;
        !matches!(u, U8(0u8) | U16(0u16) | U32(0u32) | U64(0u64))
    }
}

impl CastFrom<UInt> for u8 {
    fn cast_from(u: UInt) -> u8 {
        use UInt::*;
        match u {
            U8(u) => u,
            U16(u) => u as u8,
            U32(u) => u as u8,
            U64(u) => u as u8,
        }
    }
}

impl CastFrom<UInt> for u16 {
    fn cast_from(u: UInt) -> u16 {
        use UInt::*;
        match u {
            U8(u) => u as u16,
            U16(u) => u,
            U32(u) => u as u16,
            U64(u) => u as u16,
        }
    }
}

impl CastFrom<UInt> for u32 {
    fn cast_from(u: UInt) -> u32 {
        use UInt::*;
        match u {
            U8(u) => u as u32,
            U16(u) => u as u32,
            U32(u) => u,
            U64(u) => u as u32,
        }
    }
}

impl Eq for UInt {}

impl PartialEq for UInt {
    fn eq(&self, other: &UInt) -> bool {
        match (self, other) {
            (Self::U8(l), Self::U8(r)) => l.eq(r),
            (Self::U16(l), Self::U16(r)) => l.eq(r),
            (Self::U32(l), Self::U32(r)) => l.eq(r),
            (Self::U64(l), Self::U64(r)) => l.eq(r),
            (l, r) => u64::from(*l).eq(&u64::from(*r)),
        }
    }
}

impl Hash for UInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        u64::from(*self).hash(state);
    }
}

impl PartialOrd for UInt {
    fn partial_cmp(&self, other: &UInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UInt {
    fn cmp(&self, other: &UInt) -> Ordering {
        match (self, other) {
            (Self::U8(l), Self::U8(r)) => l.cmp(r),
            (Self::U16(l), Self::U16(r)) => l.cmp(r),
            (Self::U32(l), Self::U32(r)) => l.cmp(r),
            (Self::U64(l), Self::U64(r)) => l.cmp(r),
            (l, r) => u64::from(*l).cmp(&u64::from(*r)),
        }
    }
}

impl Default for UInt {
    fn default() -> UInt {
        UInt::U8(u8::default())
    }
}

impl From<Boolean> for UInt {
    fn from(b: Boolean) -> UInt {
        match b {
            Boolean(true) => UInt::U8(1),
            Boolean(false) => UInt::U8(0),
        }
    }
}

impl From<u8> for UInt {
    fn from(u: u8) -> UInt {
        UInt::U8(u)
    }
}

impl From<u16> for UInt {
    fn from(u: u16) -> UInt {
        UInt::U16(u)
    }
}

impl From<u32> for UInt {
    fn from(u: u32) -> UInt {
        UInt::U32(u)
    }
}

impl From<u64> for UInt {
    fn from(u: u64) -> UInt {
        UInt::U64(u)
    }
}

impl From<UInt> for u64 {
    fn from(u: UInt) -> u64 {
        match u {
            UInt::U64(u) => u,
            UInt::U32(u) => u as u64,
            UInt::U16(u) => u as u64,
            UInt::U8(u) => u as u64,
        }
    }
}

impl CastFrom<UInt> for usize {
    fn cast_from(u: UInt) -> usize {
        match u {
            UInt::U64(u) => u as usize,
            UInt::U32(u) => u as usize,
            UInt::U16(u) => u as usize,
            UInt::U8(u) => u as usize,
        }
    }
}

impl FromStr for UInt {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        u64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

fmt_debug!(UInt);

impl fmt::Display for UInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UInt::U8(u) => fmt::Display::fmt(u, f),
            UInt::U16(u) => fmt::Display::fmt(u, f),
            UInt::U32(u) => fmt::Display::fmt(u, f),
            UInt::U64(u) => fmt::Display::fmt(u, f),
        }
    }
}

macro_rules! checked_real_functions {
    ($($try:ident => $name:ident, |$x:ident| $invalid:expr, $zero:expr;)*) => {$ (
        fn $try(self) -> Result<Self::Out, Error> {
            let $x = f64::from(self);
            let finite = checked::finite(self);
            if finite && $invalid { return Err(checked::error(ErrorKind::Domain, stringify!($name))); }
            let result = checked::math::$name(self);
            checked::float_result(finite, result, $zero, ErrorKind::Underflow, stringify!($name))
        }
    )*};
}

impl Trigonometry for Float {
    type Out = Self;
    ordinary_methods!(float; asin => Out, sin => Out, sinh => Out, asinh => Out, acos => Out, cos => Out, cosh => Out, acosh => Out, atan => Out, tan => Out, tanh => Out, atanh => Out);
    checked_real_functions! {
        try_asin => asin, |x| x.abs() > 1.0, x == 0.0;
        try_sin => sin, |x| false, x == 0.0;
        try_sinh => sinh, |x| false, x == 0.0;
        try_asinh => asinh, |x| false, x == 0.0;
        try_acos => acos, |x| x.abs() > 1.0, x == 1.0;
        try_cos => cos, |_x| false, false;
        try_cosh => cosh, |_x| false, false;
        try_acosh => acosh, |x| x < 1.0, x == 1.0;
        try_atan => atan, |x| false, x == 0.0;
        try_tan => tan, |x| false, x == 0.0;
        try_tanh => tanh, |x| false, x == 0.0;
        try_atanh => atanh, |x| x.abs() >= 1.0, x == 0.0;
    }
}

impl Trigonometry for Complex {
    ordinary_methods!(complex; asin => Out, sin => Out, sinh => Out, asinh => Out, acos => Out, cos => Out, cosh => Out, acosh => Out, atan => Out, tan => Out, tanh => Out, atanh => Out);
    type Out = Self;

    fn try_asin(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::asin_axis(self, Float::try_asin)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_asinh()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex asin input",
        ))
    }

    fn try_sin(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::sin_axis(self, Float::try_sin)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_sinh()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex sin input",
        ))
    }

    fn try_sinh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::sinh_axis(self, Float::try_sinh)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_sin()?;
            return Ok(crate::native::parts(
                self.re() * checked::math::cos(self.im()),
                component,
            ));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex sinh input",
        ))
    }

    fn try_asinh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::asinh_axis(self, Float::try_asinh)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        let y = f64::from(self.im());
        if checked::complex_finite(self) && x == 0.0 && (y.abs() <= 1.0) {
            let component = self.im().try_asin()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex asinh input",
        ))
    }

    fn try_acos(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::acos_axis(self, Float::try_acos)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_asinh()?;
            return Ok(crate::native::parts(
                checked::math::acos(self.re()),
                crate::native::negate(component),
            ));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex acos input",
        ))
    }

    fn try_cos(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::cos_axis(self, Float::try_cos)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_cosh()?;
            return Ok(crate::native::parts(
                component,
                crate::native::negate(self.re() * self.im()),
            ));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex cos input",
        ))
    }

    fn try_cosh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::cosh_axis(self, Float::try_cosh)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_cos()?;
            return Ok(crate::native::parts(
                component,
                self.re() * checked::math::sin(self.im()),
            ));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex cosh input",
        ))
    }

    fn try_acosh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::acosh_axis(self, Float::try_acosh)? {
            return Ok(result);
        }

        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex acosh input",
        ))
    }

    fn try_atan(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::atan_axis(self, Float::try_atan)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        let y = f64::from(self.im());
        if checked::complex_finite(self) && x == 0.0 && y.abs() == 1.0 {
            return Err(checked::error(ErrorKind::Domain, "atan"));
        }
        if checked::complex_finite(self) && x == 0.0 && (y.abs() < 1.0) {
            let component = self.im().try_atanh()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex atan input",
        ))
    }

    fn try_tan(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::tan_axis(self, Float::try_tan)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_tanh()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex tan input",
        ))
    }

    fn try_tanh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::tanh_axis(self, Float::try_tanh)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_tan()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex tanh input",
        ))
    }

    fn try_atanh(self) -> Result<Self::Out, Error> {
        if let Some(result) = checked::math::atanh_axis(self, Float::try_atanh)? {
            return Ok(result);
        }

        let x = f64::from(self.re());
        let y = f64::from(self.im());
        if checked::complex_finite(self) && y == 0.0 && x.abs() == 1.0 {
            return Err(checked::error(ErrorKind::Domain, "atanh"));
        }
        if checked::complex_finite(self) && x == 0.0 {
            let component = self.im().try_atan()?;
            return Ok(crate::native::parts(self.re(), component));
        }
        Err(checked::error(
            ErrorKind::Unsupported,
            "uncertified complex atanh input",
        ))
    }
}

impl TryFrom<UInt> for Int {
    type Error = Error;
    fn try_from(value: UInt) -> Result<Self, Error> {
        IntType::Int.try_cast(value.into())
    }
}

impl TryFrom<UInt> for usize {
    type Error = Error;
    fn try_from(value: UInt) -> Result<Self, Error> {
        usize::try_from(u64::from(value))
            .map_err(|_| checked::error(ErrorKind::Overflow, "usize conversion"))
    }
}

macro_rules! real_trigonometry {
    ($ty:ident; $($name:ident => $try:ident),*) => {
        impl Trigonometry for $ty {
            type Out = Float;
            $(fn $name(self) -> Self::Out { Float::from(self).$name() }
              fn $try(self) -> Result<Self::Out, Error> { Float::from(self).$try() })*
        }
    };
}
real_trigonometry!(Boolean; asin => try_asin, sin => try_sin, sinh => try_sinh, asinh => try_asinh, acos => try_acos, cos => try_cos, cosh => try_cosh, acosh => try_acosh, atan => try_atan, tan => try_tan, tanh => try_tanh, atanh => try_atanh);
real_trigonometry!(Int; asin => try_asin, sin => try_sin, sinh => try_sinh, asinh => try_asinh, acos => try_acos, cos => try_cos, cosh => try_cosh, acosh => try_acosh, atan => try_atan, tan => try_tan, tanh => try_tanh, atanh => try_atanh);
real_trigonometry!(UInt; asin => try_asin, sin => try_sin, sinh => try_sinh, asinh => try_asinh, acos => try_acos, cos => try_cos, cosh => try_cosh, acosh => try_acosh, atan => try_atan, tan => try_tan, tanh => try_tanh, atanh => try_atanh);
