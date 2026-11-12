//! Interoperability delegates to the ordinary, checked, and wrapping owners.

use num::traits::{
    Bounded, CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, ConstOne, ConstZero, One, Signed,
    Unsigned, WrappingAdd, WrappingMul, WrappingSub, Zero,
};

use crate::{_Complex, Boolean, Complex, Float, Int, Number, NumberClass, NumberInstance, UInt};

mod conversions;

macro_rules! const_identities {
    ($($ty:ty => ($zero:expr, $one:expr);)+) => {$(
        impl ConstZero for $ty {
            const ZERO: Self = $zero;
        }

        impl ConstOne for $ty {
            const ONE: Self = $one;
        }
    )+};
}

const_identities! {
    Int => (Int::I16(0), Int::I16(1));
    UInt => (UInt::U8(0), UInt::U8(1));
    Float => (Float::F32(0.), Float::F32(1.));
    Complex => (
        Complex::C32(_Complex::new(0., 0.)),
        Complex::C32(_Complex::new(1., 0.))
    );
    Number => (
        Number::Bool(<Boolean as ConstZero>::ZERO),
        Number::Bool(<Boolean as ConstOne>::ONE)
    );
}

macro_rules! identities {
    ($($ty:ty),+ $(,)?) => {$(
        impl Zero for $ty {
            fn zero() -> Self {
                <Self as ConstZero>::ZERO
            }

            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }

            fn set_zero(&mut self) {
                *self = self.class().zero();
            }
        }

        impl One for $ty {
            fn one() -> Self {
                <Self as ConstOne>::ONE
            }

            fn is_one(&self) -> bool {
                *self == Self::one()
            }

            fn set_one(&mut self) {
                *self = self.class().one();
            }
        }
    )+};
}

macro_rules! checked {
    ($trait:ident, $method:ident, $owner:ident; $($ty:ty),+ $(,)?) => {$(
        impl $trait for $ty {
            fn $method(&self, other: &Self) -> Option<Self> {
                <Self as NumberInstance>::$owner(*self, *other).ok()
            }
        }
    )+};
}

macro_rules! wrapping {
    ($trait:ident, $method:ident; $($ty:ty),+ $(,)?) => {$(
        impl $trait for $ty {
            fn $method(&self, other: &Self) -> Self {
                Self::$method(*self, *other)
            }
        }
    )+};
}

identities!(Boolean, Int, UInt, Float, Complex, Number);

checked!(CheckedAdd, checked_add, try_add; Boolean, Int, UInt, Float, Complex, Number);
checked!(CheckedSub, checked_sub, try_sub; Boolean, Int, UInt, Float, Complex, Number);
checked!(CheckedMul, checked_mul, try_mul; Boolean, Int, UInt, Float, Complex, Number);
checked!(CheckedDiv, checked_div, try_div; Boolean, Int, UInt, Float, Complex, Number);

wrapping!(WrappingAdd, wrapping_add; Int, UInt, Number);
wrapping!(WrappingSub, wrapping_sub; Int, UInt, Number);
wrapping!(WrappingMul, wrapping_mul; Int, UInt, Number);

impl Unsigned for UInt {}

impl Signed for Float {
    fn abs(&self) -> Self {
        NumberInstance::abs(*self)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        let dtype = Ord::max(self.class(), other.class());
        match (self.into_type(dtype), other.into_type(dtype)) {
            (Self::F32(left), Self::F32(right)) => Self::F32(Signed::abs_sub(&left, &right)),
            (Self::F64(left), Self::F64(right)) => Self::F64(Signed::abs_sub(&left, &right)),
            _ => unreachable!("prepared floating widths"),
        }
    }

    fn signum(&self) -> Self {
        match self {
            Self::F32(v) => Self::F32(Signed::signum(v)),
            Self::F64(v) => Self::F64(Signed::signum(v)),
        }
    }

    fn is_positive(&self) -> bool {
        match self {
            Self::F32(v) => Signed::is_positive(v),
            Self::F64(v) => Signed::is_positive(v),
        }
    }

    fn is_negative(&self) -> bool {
        match self {
            Self::F32(v) => Signed::is_negative(v),
            Self::F64(v) => Signed::is_negative(v),
        }
    }
}

macro_rules! bounds {
    ($($ty:ty => ($min:expr, $max:expr);)+) => {$(
        impl Bounded for $ty {
            fn min_value() -> Self { $min }

            fn max_value() -> Self { $max }
        }
    )+};
}

bounds! {
    Boolean => (<Boolean as ConstZero>::ZERO, <Boolean as ConstOne>::ONE);
    Int => (Int::I64(i64::MIN), Int::I64(i64::MAX));
    UInt => (UInt::U64(0), UInt::U64(u64::MAX));
    Float => (Float::F64(f64::MIN), Float::F64(f64::MAX));
}
