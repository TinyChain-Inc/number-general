//! Unary numerical behavior; ecosystem traits delegate here.

use std::ops::Neg;

use num::traits::{CheckedNeg, Inv, WrappingNeg};

use crate::{
    checked, native, wrapping, Complex, Error, ErrorKind, Float, Int, Number, NumberClass,
    NumberInstance, UInt,
};

macro_rules! signed_negation {
    ($($variant:ident),+) => {
        impl Neg for Int {
            type Output = Self;

            fn neg(self) -> Self {
                match self { $(Self::$variant(value) => Self::$variant(-value),)+ }
            }
        }

        impl Int {
            /// Negate without changing width, rejecting the signed minimum.
            pub fn try_neg(self) -> Result<Self, Error> {
                checked::signed(-i128::from(i64::from(self)), self.class())
            }

            /// Negate modulo this integer's width, including its signed minimum.
            pub fn wrapping_neg(self) -> Self {
                match self { $(Self::$variant(value) => Self::$variant(value.wrapping_neg()),)+ }
            }
        }
    };
}

signed_negation!(I8, I16, I32, I64);

impl UInt {
    /// Zero has an unsigned negation; other values return Underflow.
    pub fn try_neg(self) -> Result<Self, Error> {
        if u64::from(self) == 0 {
            Ok(self)
        } else {
            Err(checked::error(ErrorKind::Underflow, "unsigned negation"))
        }
    }

    /// Negate modulo this unsigned integer's width.
    pub fn wrapping_neg(self) -> Self {
        match self {
            Self::U8(v) => Self::U8(v.wrapping_neg()),
            Self::U16(v) => Self::U16(v.wrapping_neg()),
            Self::U32(v) => Self::U32(v.wrapping_neg()),
            Self::U64(v) => Self::U64(v.wrapping_neg()),
        }
    }
}

impl Neg for Float {
    type Output = Self;

    fn neg(self) -> Self {
        native::negate(self)
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::C32(v) => Self::C32(-v),
            Self::C64(v) => Self::C64(-v),
        }
    }
}

impl Neg for Number {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            Self::Int(v) => (-v).into(),
            Self::Float(v) => (-v).into(),
            Self::Complex(v) => (-v).into(),
            Self::Bool(_) | Self::UInt(_) => {
                self.try_neg().expect("unrepresentable signed negation")
            }
        }
    }
}

impl Number {
    /// Negate using existing signed-family mappings for Boolean/unsigned values.
    /// Narrow only after evaluating the mathematical negation.
    pub fn try_neg(self) -> Result<Self, Error> {
        match self {
            Self::Int(v) => v.try_neg().map(Self::from),
            Self::Float(v) => v.try_neg().map(Self::from),
            Self::Complex(v) => v.try_neg().map(Self::from),
            Self::Bool(_) | Self::UInt(_) => {
                checked::signed(-self.integer(), checked::signed_type(self)).map(Self::from)
            }
        }
    }

    /// Negate into the ordinarily selected result family with integer wrapping.
    pub fn wrapping_neg(self) -> Self {
        match self {
            Self::Int(v) => v.wrapping_neg().into(),
            Self::Float(v) => (-v).into(),
            Self::Complex(v) => (-v).into(),
            Self::Bool(_) | Self::UInt(_) => {
                wrapping::signed(-self.integer(), checked::signed_type(self)).into()
            }
        }
    }

    fn reciprocal_value(self) -> Self {
        match self {
            Self::Bool(v) => Float::from(v).into(),
            Self::Int(v) => Float::from(v).into(),
            Self::UInt(v) => Float::from(v).into(),
            Self::Float(_) | Self::Complex(_) => self,
        }
    }

    /// Reciprocal after ordinary floating-family promotion of integer/Boolean inputs.
    pub fn recip(self) -> Self {
        let value = self.reciprocal_value();
        value.class().one() / value
    }

    /// Checked reciprocal with the same promotion as ordinary reciprocal.
    pub fn try_recip(self) -> Result<Self, Error> {
        let value = self.reciprocal_value();
        value.class().one().try_div(value)
    }
}

macro_rules! floating_unary {
    ($($ty:ty),+) => {$(
        impl $ty {
            /// Direct component negation preserves width, signed zeros and nonfinite inputs.
            pub fn try_neg(self) -> Result<Self, Error> {
                Ok(-self)
            }

            /// Reciprocal through ordinary division by a same-dtype one.
            pub fn recip(self) -> Self {
                self.class().one() / self
            }

            /// Reciprocal through checked division by a same-dtype one.
            pub fn try_recip(self) -> Result<Self, Error> {
                self.class().one().try_div(self)
            }
        }
    )+};
}

floating_unary!(Float, Complex);

macro_rules! checked_negation {
    ($($ty:ty),+) => {$(
        impl CheckedNeg for $ty {
            fn checked_neg(&self) -> Option<Self> { Self::try_neg(*self).ok() }
        }
    )+};
}

checked_negation!(Int, UInt, Float, Complex, Number);

macro_rules! wrapping_negation {
    ($($ty:ty),+) => {$(
        impl WrappingNeg for $ty {
            fn wrapping_neg(&self) -> Self { Self::wrapping_neg(*self) }
        }
    )+};
}

wrapping_negation!(Int, UInt, Number);

macro_rules! inverse {
    ($($ty:ty),+) => {$(
        impl Inv for $ty {
            type Output = Self;

            fn inv(self) -> Self { self.recip() }
        }
    )+};
}

inverse!(Float, Complex, Number);
