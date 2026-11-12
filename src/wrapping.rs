//! Explicit integer wrapping with the ordinary scalar promotion rules.

use num::traits::Pow;

use crate::{
    checked, native, Int, IntType, Number, NumberClass, NumberInstance, NumberType, UInt, UIntType,
};

macro_rules! typed_binary {
    ($name:ident, $ty:ident; $($variant:ident),+) => {
        #[doc = concat!("Promote operand widths, then apply `", stringify!($name), "`. Integer zero divisors panic.")]
        pub fn $name(self, other: Self) -> Self {
            let dtype = Ord::max(self.class(), other.class());
            match (self.into_type(dtype), other.into_type(dtype)) {
                $(($ty::$variant(left), $ty::$variant(right)) => Self::$variant(left.$name(right)),)+
                _ => unreachable!("promoted integer widths"),
            }
        }
    };
}

macro_rules! integer_methods {
    ($ty:ident; $($variant:ident),+) => {
        impl $ty {
            typed_binary!(wrapping_add, $ty; $($variant),+);
            typed_binary!(wrapping_sub, $ty; $($variant),+);
            typed_binary!(wrapping_mul, $ty; $($variant),+);
            typed_binary!(wrapping_div, $ty; $($variant),+);
            typed_binary!(wrapping_rem, $ty; $($variant),+);

            /// Absolute value with wrapping at the selected integer width.
            pub fn wrapping_abs(self) -> Self {
                match self {
                    $(Self::$variant(value) => Self::$variant(
                        if (value as i128) < 0 { value.wrapping_neg() } else { value }
                    ),)+
                }
            }

            /// Raise to a full-width nonnegative exponent, wrapping each multiplication.
            pub fn wrapping_pow(self, exponent: u64) -> Self {
                native::power(self, exponent, self.class().one(), |a, b| Ok(a.wrapping_mul(b)))
                    .expect("wrapping integer multiplication is infallible")
            }
        }
    };
}

integer_methods!(Int; I8, I16, I32, I64);
integer_methods!(UInt; U8, U16, U32, U64);

pub(super) fn signed(value: i128, dtype: IntType) -> Int {
    match dtype {
        IntType::I8 => Int::I8(value as i8),
        IntType::I16 => Int::I16(value as i16),
        IntType::I32 => Int::I32(value as i32),
        IntType::I64 => Int::I64(value as i64),
        IntType::Int => unreachable!("resolved signed width"),
    }
}

fn unsigned(value: u128, dtype: UIntType) -> UInt {
    match dtype {
        UIntType::U8 => UInt::U8(value as u8),
        UIntType::U16 => UInt::U16(value as u16),
        UIntType::U32 => UInt::U32(value as u32),
        UIntType::U64 => UInt::U64(value as u64),
        UIntType::UInt => unreachable!("resolved unsigned width"),
    }
}

// Signed and mixed-family operands are at most 64 bits, with at least one signed
// operand, so their products fit i128. Unsigned products fit u128. Narrow only the
// mathematical result; unsigned subtraction uses modular u128 arithmetic first.
macro_rules! number_binary {
    ($($name:ident, $op:tt, $unsigned:ident;)*) => {
        impl Number {$(
            #[doc = concat!("Use ordinary numeric promotion and `", stringify!($name), "` for integer results. Floating/complex results use IEEE behavior; integer zero divisors panic.")]
            pub fn $name(self, other: Self) -> Self {
                match checked::promote(self, other) {
                    NumberType::Int(dtype) => signed(self.integer() $op other.integer(), dtype).into(),
                    NumberType::UInt(dtype) => unsigned(
                        (self.integer() as u128).$unsigned(other.integer() as u128), dtype
                    ).into(),
                    _ => self $op other,
                }
            }
        )*}
    };
}

number_binary! {
    wrapping_add, +, wrapping_add;
    wrapping_sub, -, wrapping_sub;
    wrapping_mul, *, wrapping_mul;
    wrapping_div, /, wrapping_div;
    wrapping_rem, %, wrapping_rem;
}

impl Number {
    /// Absolute value, wrapping integer minima and using ordinary floating/complex behavior.
    pub fn wrapping_abs(self) -> Self {
        match self {
            Self::Int(value) => value.wrapping_abs().into(),
            Self::UInt(value) => value.wrapping_abs().into(),
            other => other.abs(),
        }
    }

    /// Integer powers wrap; negative or floating/complex exponents use ordinary promotion.
    pub fn wrapping_pow(self, exponent: Self) -> Self {
        if self.is_int() && !native::promotes_power(exponent) {
            let exponent_value = native::integer_exponent(exponent)
                .expect("nonnegative integer exponent")
                .magnitude;
            match self {
                Self::Int(value) => value.wrapping_pow(exponent_value).into(),
                Self::UInt(value) => value.wrapping_pow(exponent_value).into(),
                other => other.pow(exponent),
            }
        } else {
            self.pow(exponent)
        }
    }
}

macro_rules! aggregates {
    ($($ty:ident),+) => {$(
        impl $ty {
            /// Sum left-to-right with ordinary promotion and explicit integer wrapping.
            /// Empty input retains the ordinary sum identity.
            pub fn wrapping_sum<I: Iterator<Item = Self>>(mut values: I) -> Self {
                let Some(first) = values.next() else {
                    return Self::default().class().zero();
                };
                values.fold(first, Self::wrapping_add)
            }

            /// Multiply left-to-right without skipping later promotion after zero.
            /// Empty input retains the ordinary product identity.
            pub fn wrapping_product<I: Iterator<Item = Self>>(mut values: I) -> Self {
                let Some(first) = values.next() else {
                    return Self::default().class().one();
                };
                values.fold(first, Self::wrapping_mul)
            }
        }
    )+};
}

aggregates!(Int, UInt, Number);
