//! Primitive conversion and radix parsing delegate to their native owners.

use num::traits::{FromPrimitive, Num, ToPrimitive};

use crate::{_Complex, Boolean, Complex, Error, ErrorKind, Float, Int, Number, UInt};

macro_rules! to_primitives {
    ($($method:ident -> $out:ty),+ $(,)?) => {
        impl ToPrimitive for Int {$(
            fn $method(&self) -> Option<$out> {
                match self {
                    Self::I8(v) => ToPrimitive::$method(v),
                    Self::I16(v) => ToPrimitive::$method(v),
                    Self::I32(v) => ToPrimitive::$method(v),
                    Self::I64(v) => ToPrimitive::$method(v),
                }
            }
        )+}

        impl ToPrimitive for UInt {$(
            fn $method(&self) -> Option<$out> {
                match self {
                    Self::U8(v) => ToPrimitive::$method(v),
                    Self::U16(v) => ToPrimitive::$method(v),
                    Self::U32(v) => ToPrimitive::$method(v),
                    Self::U64(v) => ToPrimitive::$method(v),
                }
            }
        )+}

        impl ToPrimitive for Float {$(
            fn $method(&self) -> Option<$out> {
                match self {
                    Self::F32(v) => ToPrimitive::$method(v),
                    Self::F64(v) => ToPrimitive::$method(v),
                }
            }
        )+}

        impl ToPrimitive for Boolean {$(
            fn $method(&self) -> Option<$out> {
                ToPrimitive::$method(&u8::from(bool::from(*self)))
            }
        )+}

        impl ToPrimitive for Complex {$(
            fn $method(&self) -> Option<$out> {
                match self {
                    Self::C32(v) if v.im == 0. => ToPrimitive::$method(&v.re),
                    Self::C64(v) if v.im == 0. => ToPrimitive::$method(&v.re),
                    _ => None,
                }
            }
        )+}

        impl ToPrimitive for Number {$(
            fn $method(&self) -> Option<$out> {
                match self {
                    Self::Bool(v) => ToPrimitive::$method(v),
                    Self::Int(v) => ToPrimitive::$method(v),
                    Self::UInt(v) => ToPrimitive::$method(v),
                    Self::Float(v) => ToPrimitive::$method(v),
                    Self::Complex(v) => ToPrimitive::$method(v),
                }
            }
        )+}
    };
}

to_primitives! {
    to_i8 -> i8, to_i16 -> i16, to_i32 -> i32, to_i64 -> i64, to_i128 -> i128,
    to_isize -> isize,
    to_u8 -> u8, to_u16 -> u16, to_u32 -> u32, to_u64 -> u64, to_u128 -> u128,
    to_usize -> usize,
    to_f32 -> f32, to_f64 -> f64,
}

macro_rules! pointer_imports {
    () => {
        fn from_isize(value: isize) -> Option<Self> {
            #[cfg(target_pointer_width = "16")]
            return Self::from_i16(value as i16);
            #[cfg(target_pointer_width = "32")]
            return Self::from_i32(value as i32);
            #[cfg(target_pointer_width = "64")]
            return Self::from_i64(value as i64);
        }

        fn from_usize(value: usize) -> Option<Self> {
            #[cfg(target_pointer_width = "16")]
            return Self::from_u16(value as u16);
            #[cfg(target_pointer_width = "32")]
            return Self::from_u32(value as u32);
            #[cfg(target_pointer_width = "64")]
            return Self::from_u64(value as u64);
        }
    };
}

macro_rules! from_primitives {
    ($value:ident; $(
        $method:ident($source:ty) => ($signed:ident, $unsigned:ident, $float:ident, $dynamic:expr);
    )+) => {
        impl FromPrimitive for Int {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                from_primitives!(@signed $signed, $value)
            })+
        }

        impl FromPrimitive for UInt {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                from_primitives!(@unsigned $unsigned, $value)
            })+
        }

        impl FromPrimitive for Float {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                from_primitives!(@float $float, $value)
            })+
        }

        impl FromPrimitive for Complex {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                ToPrimitive::to_f64(&$value).map(|re| Self::C64(_Complex::new(re, 0.)))
            })+
        }

        impl FromPrimitive for Boolean {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                if $value == 0 as $source {
                    Some(Self::from(false))
                } else if $value == 1 as $source {
                    Some(Self::from(true))
                } else {
                    None
                }
            })+
        }

        impl FromPrimitive for Number {
            pointer_imports!();

            $(fn $method($value: $source) -> Option<Self> {
                $dynamic
            })+
        }
    };
    (@signed I8, $v:ident) => { Some(Int::I8($v)) };
    (@signed I16, $v:ident) => { Some(Int::I16($v)) };
    (@signed I32, $v:ident) => { Some(Int::I32($v)) };
    (@signed I64, $v:ident) => { ToPrimitive::to_i64(&$v).map(Int::I64) };
    (@unsigned U8, $v:ident) => { Some(UInt::U8($v)) };
    (@unsigned U16, $v:ident) => { Some(UInt::U16($v)) };
    (@unsigned U32, $v:ident) => { Some(UInt::U32($v)) };
    (@unsigned U64, $v:ident) => { ToPrimitive::to_u64(&$v).map(UInt::U64) };
    (@float F32, $v:ident) => { Some(Float::F32($v)) };
    (@float F64, $v:ident) => { ToPrimitive::to_f64(&$v).map(Float::F64) };
}

from_primitives! { value;
    from_i8(i8) => (I8, U64, F64, Some(Number::from(value)));
    from_i16(i16) => (I16, U64, F64, Some(Number::from(value)));
    from_i32(i32) => (I32, U64, F64, Some(Number::from(value)));
    from_i64(i64) => (I64, U64, F64, Some(Number::from(value)));
    from_i128(i128) => (I64, U64, F64,
        value.to_i64().map(Number::from).or_else(|| value.to_u64().map(Number::from)));
    from_u8(u8) => (I64, U8, F64, Some(Number::from(value)));
    from_u16(u16) => (I64, U16, F64, Some(Number::from(value)));
    from_u32(u32) => (I64, U32, F64, Some(Number::from(value)));
    from_u64(u64) => (I64, U64, F64, Some(Number::from(value)));
    from_u128(u128) => (I64, U64, F64, value.to_u64().map(Number::from));
    from_f32(f32) => (I64, U64, F32, Some(Number::from(value)));
    from_f64(f64) => (I64, U64, F64, Some(Number::from(value)));
}

macro_rules! decimal_num {
    ($($ty:ty),+) => {$(
        impl Num for $ty {
            type FromStrRadixErr = Error;

            fn from_str_radix(source: &str, radix: u32) -> Result<Self, Error> {
                validate_radix(radix)?;
                if radix == 10 {
                    source.parse()
                } else {
                    Err(Error::with_kind(
                        ErrorKind::Unsupported,
                        "nondecimal parsing requires an integer family",
                    ))
                }
            }
        }
    )+};
}

decimal_num!(Boolean, Float, Complex);

fn validate_radix(radix: u32) -> Result<(), Error> {
    if (2..=36).contains(&radix) {
        Ok(())
    } else {
        Err(Error::with_kind(
            ErrorKind::InvalidInput,
            "radix must be between 2 and 36",
        ))
    }
}

macro_rules! integer_num {
    ($($ty:ty => $primitive:ty),+) => {$(
        impl Num for $ty {
            type FromStrRadixErr = Error;

            fn from_str_radix(source: &str, radix: u32) -> Result<Self, Error> {
                validate_radix(radix)?;
                if radix == 10 {
                    source.parse()
                } else {
                    <$primitive>::from_str_radix(source, radix)
                        .map(Self::from)
                        .map_err(Error::new)
                }
            }
        }
    )+};
}

integer_num!(Int => i64, UInt => u64);

impl Num for Number {
    type FromStrRadixErr = Error;

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Error> {
        validate_radix(radix)?;
        if radix == 10 {
            return source.parse();
        }

        let source = source.trim();
        Int::from_str_radix(source, radix)
            .map(Self::from)
            .or_else(|_| UInt::from_str_radix(source, radix).map(Self::from))
    }
}
