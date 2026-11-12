use num::traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, ConstOne, ConstZero, One, WrappingAdd,
    WrappingMul, WrappingSub, Zero,
};

use crate::{Boolean, Complex, Float, Int, IntType, Number, NumberInstance, UInt};

fn identities<T: NumberInstance + PartialEq + Zero + One + ConstZero + ConstOne>(value: T)
where
    T::Class: std::fmt::Debug,
{
    let zero = <T as Zero>::zero();
    let one = <T as One>::one();
    assert_eq!(zero, <T as ConstZero>::ZERO);
    assert_eq!(one, <T as ConstOne>::ONE);
    assert_eq!(zero.class(), <T as ConstZero>::ZERO.class());
    assert_eq!(one.class(), <T as ConstOne>::ONE.class());
    assert_eq!(zero.class(), T::default().class());
    assert_eq!(one.class(), T::default().class());
    assert!(Zero::is_zero(&zero));
    assert!(!One::is_one(&zero));
    assert!(One::is_one(&one));
    assert!(!Zero::is_zero(&one));

    let mut changed = value;
    Zero::set_zero(&mut changed);
    assert_eq!(changed.class(), value.class());
    assert!(Zero::is_zero(&changed));
    One::set_one(&mut changed);
    assert_eq!(changed.class(), value.class());
    assert!(One::is_one(&changed));
}

#[test]
fn identity_constructors_and_setters_preserve_their_contracts() {
    macro_rules! constant_context {
        ($($ty:ty),+) => {$( {
            const ZERO: $ty = <$ty as ConstZero>::ZERO;
            const ONE: $ty = <$ty as ConstOne>::ONE;
            assert!(Zero::is_zero(&ZERO));
            assert!(One::is_one(&ONE));
            assert_eq!(ZERO.class(), <$ty>::default().class());
            assert_eq!(ONE.class(), <$ty>::default().class());
        } )+};
    }
    constant_context!(Boolean, Int, UInt, Float, Complex, Number);
    identities(Boolean::from(true));

    macro_rules! integer {
        ($($ty:ty),+) => {$(
            identities(Int::from(3 as $ty));
            identities(Number::from(3 as $ty));
            assert!(Zero::is_zero(&Int::from(0 as $ty)));
            assert!(One::is_one(&Int::from(1 as $ty)));
        )+};
    }
    integer!(i8, i16, i32, i64);

    macro_rules! unsigned {
        ($($ty:ty),+) => {$(
            identities(UInt::from(3 as $ty));
            identities(Number::from(3 as $ty));
            assert!(Zero::is_zero(&UInt::from(0 as $ty)));
            assert!(One::is_one(&UInt::from(1 as $ty)));
        )+};
    }
    unsigned!(u8, u16, u32, u64);
    for value in [Float::from(3f32), Float::from(3f64)] {
        identities(value);
        identities(Number::from(value));
    }
    for value in [Complex::from([2f32, -3.]), Complex::from([2f64, -3.])] {
        identities(value);
        identities(Number::from(value));
    }
    identities(Number::from(true));

    assert_eq!(<Int as Zero>::zero(), Int::from(0i16));
    assert_eq!(<UInt as One>::one().class(), UInt::from(1u8).class());
    assert_eq!(<Float as One>::one().class(), Float::from(1f32).class());
    assert_eq!(
        <Complex as Zero>::zero().class(),
        Complex::from([0f32, 0.]).class()
    );
    assert_eq!(
        <Number as Zero>::zero().class(),
        Number::from(false).class()
    );
    let widened = CheckedAdd::checked_add(&Int::from(7i8), &<Int as Zero>::zero()).unwrap();
    assert_eq!(widened.class(), IntType::I16);
    assert_eq!(widened, Int::from(7i16));
}

fn checked<T>(left: T, right: T)
where
    T: NumberInstance + PartialEq + CheckedAdd + CheckedSub + CheckedMul + CheckedDiv,
    T::Class: std::fmt::Debug,
{
    let before = (left, right, left.class(), right.class());

    macro_rules! operation {
        ($trait:ident, $method:ident, $owner:ident, $operator:ident, $ordinary:ident) => {
            let actual = $trait::$method(&left, &right);
            let expected = <T as NumberInstance>::$owner(left, right).ok();
            assert_eq!(actual, expected);
            assert_eq!(actual.map(|v| v.class()), expected.map(|v| v.class()));
            // Ordinary arithmetic agrees only where checked arithmetic succeeds.
            if let Some(result) = expected {
                let ordinary = std::ops::$operator::$ordinary(left, right);
                assert_eq!(ordinary, result);
                assert_eq!(ordinary.class(), result.class());
            }
        };
    }
    operation!(CheckedAdd, checked_add, try_add, Add, add);
    operation!(CheckedSub, checked_sub, try_sub, Sub, sub);
    operation!(CheckedMul, checked_mul, try_mul, Mul, mul);
    operation!(CheckedDiv, checked_div, try_div, Div, div);
    assert_eq!((left, right, left.class(), right.class()), before);
}

#[test]
fn checked_binary_traits_delegate_without_mutating_operands() {
    checked(Boolean::from(true), Boolean::from(false));

    macro_rules! widths {
        ($enum:ident; $($ty:ty),+) => {$(
            checked($enum::from(<$ty>::MAX), $enum::from(1 as $ty));
            checked($enum::from(<$ty>::MIN), $enum::from(1 as $ty));
            checked($enum::from(1 as $ty), $enum::from(0 as $ty));
            assert!(CheckedAdd::checked_add(&$enum::from(<$ty>::MAX), &$enum::from(1 as $ty)).is_none());
        )+};
    }
    widths!(Int; i8, i16, i32, i64);
    widths!(UInt; u8, u16, u32, u64);
    checked(Int::from(100i8), Int::from(200i16));
    checked(Int::from(200i16), Int::from(100i8));
    checked(UInt::from(200u8), UInt::from(1000u64));
    checked(UInt::from(1000u64), UInt::from(200u8));
    checked(Float::from(1.5f32), Float::from(2f64));
    checked(Complex::from([1f32, 2.]), Complex::from([2f64, 0.]));
    for (left, right) in [
        (Number::from(-1i8), Number::from(255u8)),
        (Number::from(255u8), Number::from(-1i8)),
        (Number::from(u64::MAX), Number::from(-1i64)),
        (Number::from(2f32), Number::from(3i64)),
        (Number::from([1f64, 2.]), Number::from(2f32)),
    ] {
        checked(left, right);
    }
    assert_eq!(
        CheckedAdd::checked_add(&Int::from(100i8), &Int::from(200i16)),
        Some(Int::from(300i16))
    );
    assert!(CheckedSub::checked_sub(&UInt::from(0u8), &UInt::from(1u8)).is_none());
    assert!(CheckedDiv::checked_div(&Int::from(i64::MIN), &Int::from(-1i64)).is_none());
    assert!(CheckedDiv::checked_div(&Boolean::from(true), &Boolean::from(false)).is_none());
}

#[test]
fn wrapping_binary_traits_preserve_widths_and_promotion() {
    macro_rules! widths {
        ($enum:ident; $($ty:ty),+) => {$(
            let max = $enum::from(<$ty>::MAX);
            let min = $enum::from(<$ty>::MIN);
            let one = $enum::from(1 as $ty);
            let two = $enum::from(2 as $ty);
            for (actual, expected) in [
                (WrappingAdd::wrapping_add(&max, &one), $enum::from(<$ty>::MAX.wrapping_add(1))),
                (WrappingSub::wrapping_sub(&min, &one), $enum::from(<$ty>::MIN.wrapping_sub(1))),
                (WrappingMul::wrapping_mul(&max, &two), $enum::from(<$ty>::MAX.wrapping_mul(2))),
            ] {
                assert_eq!(actual, expected);
                assert_eq!(actual.class(), expected.class());
            }
            for (wrapped, ordinary, checked, expected) in [
                (WrappingAdd::wrapping_add(&one, &two), one + two,
                    CheckedAdd::checked_add(&one, &two), $enum::from(3 as $ty)),
                (WrappingSub::wrapping_sub(&two, &one), two - one,
                    CheckedSub::checked_sub(&two, &one), one),
                (WrappingMul::wrapping_mul(&one, &two), one * two,
                    CheckedMul::checked_mul(&one, &two), two),
            ] {
                assert_eq!(wrapped, expected);
                assert_eq!(ordinary, expected);
                assert_eq!(checked, Some(expected));
                assert_eq!(wrapped.class(), expected.class());
                assert_eq!(ordinary.class(), expected.class());
                assert_eq!(checked.unwrap().class(), expected.class());
            }
            assert_eq!(max.class(), $enum::from(<$ty>::MAX).class());
        )+};
    }
    widths!(Int; i8, i16, i32, i64);
    widths!(UInt; u8, u16, u32, u64);
    assert_eq!(
        WrappingAdd::wrapping_add(&Int::from(100i8), &Int::from(200i16)),
        Int::from(300i16)
    );
    assert_eq!(
        WrappingAdd::wrapping_add(&Number::from(-1i8), &Number::from(255u8)),
        Number::from(254i16)
    );
    assert_eq!(
        WrappingAdd::wrapping_add(&Number::from(-1i64), &Number::from(u64::MAX)),
        Number::from(-2i64)
    );
    for (left, right) in [
        (Number::from(-1i8), Number::from(255u8)),
        (Number::from(3u8), Number::from(-2i64)),
        (Number::from(2f32), Number::from(3f64)),
        (Number::from([2f64, 1.]), Number::from(3f32)),
        (Number::from(true), Number::from(false)),
    ] {
        assert_eq!(
            WrappingAdd::wrapping_add(&left, &right),
            Number::wrapping_add(left, right)
        );
        assert_eq!(
            WrappingSub::wrapping_sub(&left, &right),
            Number::wrapping_sub(left, right)
        );
        assert_eq!(
            WrappingMul::wrapping_mul(&left, &right),
            Number::wrapping_mul(left, right)
        );
        for (wrapped, ordinary, checked) in [
            (
                WrappingAdd::wrapping_add(&left, &right),
                left + right,
                left.try_add(right),
            ),
            (
                WrappingSub::wrapping_sub(&left, &right),
                left - right,
                left.try_sub(right),
            ),
            (
                WrappingMul::wrapping_mul(&left, &right),
                left * right,
                left.try_mul(right),
            ),
        ] {
            let checked = checked.unwrap();
            assert_eq!(wrapped, ordinary);
            assert_eq!(checked, ordinary);
            assert_eq!(wrapped.class(), ordinary.class());
            assert_eq!(checked.class(), ordinary.class());
        }
    }
}

#[test]
fn power_trait_uses_ordinary_dynamic_exponents() {
    fn power<T: NumberInstance>(value: T, exponent: Number) -> T {
        value.pow(exponent)
    }
    assert_eq!(
        power(Boolean::from(false), Number::from(0u8)),
        Boolean::from(true)
    );
    assert_eq!(power(Int::from(2i8), Number::from(3f32)), Int::from(8i8));
    assert_eq!(
        power(UInt::from(2u16), Number::from(3u64)),
        UInt::from(8u16)
    );
    assert_eq!(
        power(Float::from(2f32), Number::from(3u8)),
        Float::from(8f32)
    );
    assert_eq!(
        power(Complex::from([0f64, 1.]), Number::from(2u8)),
        Complex::from([-1f64, 0.])
    );
    assert_eq!(
        power(Number::from(2i8), Number::from(-1i8)),
        Number::from(0.5f32)
    );
    let odd = power(Int::from(-1i8), Number::from(u64::MAX));
    assert_eq!(odd, Int::from(-1i8));
    assert_eq!(odd.class(), IntType::I8);
}

#[test]
fn floating_and_complex_traits_retain_checked_policy() {
    assert!(CheckedMul::checked_mul(&Float::from(f32::MAX), &Float::from(2f32)).is_none());
    assert!(
        CheckedMul::checked_mul(&Float::from(f32::from_bits(1)), &Float::from(0.5f32)).is_none()
    );
    checked(Float::from(f64::INFINITY), Float::from(1f64));
    checked(Float::from(f32::NAN), Float::from(1f32));
    checked(Complex::from([1f64, 2.]), Complex::from([0f64, 0.]));
    for value in [Float::from(-0f32), Float::from(-0f64)] {
        assert!(Zero::is_zero(&value));
        let result = CheckedMul::checked_mul(&value, &Float::from(1f32)).unwrap();
        assert!(f64::from(result).is_sign_negative());
    }
    for value in [Complex::from([-0f32, 0.]), Complex::from([0f64, -0.])] {
        assert!(Zero::is_zero(&value));
        assert!(!One::is_one(&value));
    }
    let nan = CheckedAdd::checked_add(&Float::from(f64::NAN), &Float::from(1f64)).unwrap();
    assert!(f64::from(nan).is_nan());
    let mut negative_zero = Float::from(-0f64);
    Zero::set_zero(&mut negative_zero);
    assert_eq!(f64::from(negative_zero).to_bits(), 0f64.to_bits());
}

#[test]
fn primitive_exports_match_native_conversions() {
    use num::traits::ToPrimitive;

    macro_rules! integer_targets {
        ($primitive:expr, $scalar:expr; $($method:ident),+) => {$(
            assert_eq!(ToPrimitive::$method(&$scalar), ToPrimitive::$method(&$primitive));
        )+};
    }

    macro_rules! float_targets {
        ($primitive:expr, $scalar:expr; $($method:ident),+) => {$(
            match (ToPrimitive::$method(&$scalar), ToPrimitive::$method(&$primitive)) {
                (Some(a), Some(b)) if b.is_nan() => assert!(a.is_nan()),
                (Some(a), Some(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                (None, None) => {},
                pair => panic!("conversion mismatch: {pair:?}"),
            }
        )+};
    }

    macro_rules! check {
        ($primitive:expr, $scalar:expr) => {{
            let primitive = $primitive;
            let scalar = $scalar;
            integer_targets!(primitive, scalar; to_i8, to_i16, to_i32, to_i64, to_i128, to_isize,
                to_u8, to_u16, to_u32, to_u64, to_u128, to_usize);
            float_targets!(primitive, scalar; to_f32, to_f64);
        }};
    }

    macro_rules! integers {
        ($enum:ident; $($ty:ty),+) => {$(
            for value in [<$ty>::MIN, 0, 1, <$ty>::MAX] {
                check!(value, $enum::from(value));
                check!(value, Number::from(value));
            }
        )+};
    }
    integers!(Int; i8, i16, i32, i64);
    integers!(UInt; u8, u16, u32, u64);

    macro_rules! floats {
        ($($ty:ty),+) => {$(
            for value in [
                -0. as $ty, -0.5, 1.75, <$ty>::MAX, <$ty>::INFINITY,
                <$ty>::NEG_INFINITY, <$ty>::NAN, 2.0_f64.powi(100) as $ty,
            ] {
                check!(value, Float::from(value));
                check!(value, Complex::from([value, -0.]));
                check!(value, Number::from(value));
            }
        )+};
    }
    floats!(f32, f64);
    for value in [false, true] {
        check!(u8::from(value), Boolean::from(value));
        check!(u8::from(value), Number::from(value));
    }
    for value in [Complex::from([1f64, 1.]), Complex::from([1f64, f64::NAN])] {
        assert!(value.to_i64().is_none());
        assert!(value.to_f64().is_none());
        assert!(Number::from(value).to_u128().is_none());
    }
    assert_eq!(Float::from(2.0_f64.powi(100)).to_u128(), Some(1u128 << 100));
}

#[test]
fn primitive_imports_preserve_source_or_use_wide_destinations() {
    use num::traits::{FromPrimitive, ToPrimitive};

    use crate::{ComplexType, FloatType, NumberClass, UIntType};

    macro_rules! signed {
        ($($method:ident, $export:ident, $ty:ty);+) => {$(
            for value in [<$ty>::MIN, 0, 1, <$ty>::MAX] {
                let int = Int::$method(value).unwrap();
                assert_eq!(int, Int::from(value));
                assert_eq!(int.class(), Int::from(value).class());
                let exported = int.$export().unwrap();
                assert_eq!(exported, value);
                let restored = Int::$method(exported).unwrap();
                assert_eq!(restored, int);
                assert_eq!(restored.class(), int.class());
                let dynamic = Number::$method(value).unwrap();
                assert_eq!(dynamic.class(), Number::from(value).class());
                let restored = Number::$method(dynamic.$export().unwrap()).unwrap();
                assert_eq!(restored, dynamic);
                assert_eq!(restored.class(), dynamic.class());
                assert_eq!(UInt::$method(value), value.to_u64().map(UInt::U64));
                assert_eq!(Float::$method(value).unwrap().class(), FloatType::F64);
                assert_eq!(Complex::$method(value).unwrap().class(), ComplexType::C64);
            }
        )+};
    }
    signed!(from_i8, to_i8, i8; from_i16, to_i16, i16; from_i32, to_i32, i32; from_i64, to_i64, i64);

    macro_rules! unsigned {
        ($($method:ident, $export:ident, $ty:ty);+) => {$(
            for value in [0, 1, <$ty>::MAX] {
                let uint = UInt::$method(value).unwrap();
                assert_eq!(uint, UInt::from(value));
                assert_eq!(uint.class(), UInt::from(value).class());
                let exported = uint.$export().unwrap();
                assert_eq!(exported, value);
                let restored = UInt::$method(exported).unwrap();
                assert_eq!(restored, uint);
                assert_eq!(restored.class(), uint.class());
                let dynamic = Number::$method(value).unwrap();
                let restored = Number::$method(dynamic.$export().unwrap()).unwrap();
                assert_eq!(restored, dynamic);
                assert_eq!(restored.class(), dynamic.class());
                assert_eq!(Number::$method(value).unwrap().class(), Number::from(value).class());
                assert_eq!(Int::$method(value), value.to_i64().map(Int::I64));
                let converted = Int::try_from(UInt::from(value));
                assert_eq!(converted.as_ref().ok().copied(), value.to_i64().map(Int::I64));
                if let Ok(converted) = converted { assert_eq!(converted.class(), IntType::I64); }

                assert_eq!(Float::$method(value).unwrap().class(), FloatType::F64);
            }
        )+};
    }
    unsigned!(from_u8, to_u8, u8; from_u16, to_u16, u16; from_u32, to_u32, u32; from_u64, to_u64, u64);

    macro_rules! floats {
        ($($method:ident, $export:ident, $ty:ty);+) => {$(
            for value in [-0. as $ty, -0.5, 1.5, <$ty>::MAX, <$ty>::INFINITY, <$ty>::NAN] {
                let result = Float::$method(value).unwrap();
                assert_eq!(result, Float::from(value));
                assert_eq!(result.class(), Float::from(value).class());
                let exported = result.$export().unwrap();
                if value.is_nan() {
                    assert!(exported.is_nan());
                } else {
                    assert_eq!(exported.to_bits(), value.to_bits());
                }
                let restored = Float::$method(exported).unwrap();
                assert_eq!(restored, result);
                assert_eq!(restored.class(), result.class());
                let dynamic = Number::$method(value).unwrap();
                let restored = Number::$method(dynamic.$export().unwrap()).unwrap();
                assert_eq!(restored, dynamic);
                assert_eq!(restored.class(), dynamic.class());
                assert_eq!(Int::$method(value), value.to_i64().map(Int::I64));
                assert_eq!(UInt::$method(value), value.to_u64().map(UInt::U64));
                assert_eq!(Complex::$method(value).unwrap().class(), ComplexType::C64);
                assert_eq!(Number::$method(value).unwrap().class(), Number::from(value).class());
            }
        )+};
    }
    floats!(from_f32, to_f32, f32; from_f64, to_f64, f64);
    assert_eq!(
        Number::from_i128(i64::MAX as i128 + 1),
        Some(Number::from(i64::MAX as u64 + 1))
    );
    assert_eq!(
        Number::from_i128(u64::MAX as i128),
        Some(Number::from(u64::MAX))
    );
    for value in [i128::MIN, u64::MAX as i128 + 1, i128::MAX] {
        assert!(Number::from_i128(value).is_none());
    }
    assert!(Number::from_u128(u128::MAX).is_none());
    assert_eq!(
        Number::from_u128(u64::MAX as u128),
        Some(Number::from(u64::MAX))
    );
    assert_eq!(Int::from_i128(i128::MIN), None);
    assert_eq!(UInt::from_u128(u128::MAX), None);
    assert_eq!(
        Float::from_i128(i128::MAX).unwrap(),
        Float::F64(i128::MAX as f64)
    );
    assert_eq!(
        Complex::from_u128(u128::MAX).unwrap(),
        Complex::from([u128::MAX as f64, 0.])
    );
    assert_eq!(
        Number::from_isize(1).unwrap().class(),
        if isize::BITS == 64 {
            Number::from(1i64).class()
        } else if isize::BITS == 32 {
            Number::from(1i32).class()
        } else {
            Number::from(1i16).class()
        }
    );
    assert_eq!(
        UInt::from_usize(1).unwrap().class(),
        if usize::BITS == 64 {
            UIntType::U64
        } else if usize::BITS == 32 {
            UIntType::U32
        } else {
            UIntType::U16
        }
    );
    assert_eq!(Boolean::from_f64(-0.), Some(Boolean::from(false)));
    assert_eq!(Boolean::from_i128(1), Some(Boolean::from(true)));
    assert_eq!(Boolean::from_u128(u128::MAX), None);
    assert_eq!(Boolean::from_f32(0.5), None);
    assert_eq!(Boolean::from_f64(f64::NAN), None);
    assert_eq!(Int::from_f64(1.5), Some(Int::I64(1)));
    assert!(Float::from(1.5f64).try_into_type(FloatType::F32).is_ok());
    assert!(crate::IntType::I64.try_cast(Number::from(1.5f64)).is_err());
    assert_eq!(
        Float::from_f64(-0.).unwrap().to_f64().unwrap().to_bits(),
        (-0f64).to_bits()
    );
    for value in [false, true] {
        let scalar = Boolean::from(value);
        let restored = Boolean::from_u8(scalar.to_u8().unwrap()).unwrap();
        assert_eq!(restored, scalar);
    }
    for value in [Complex::from([1.25f32, 0.]), Complex::from([-0.5f64, 0.])] {
        let restored = Complex::from_f64(value.to_f64().unwrap()).unwrap();
        assert_eq!(restored, value);
        assert_eq!(restored.class(), ComplexType::C64);
    }
    let fraction = Number::from(1.5f64);
    let truncated = Number::from_i64(fraction.to_i64().unwrap()).unwrap();
    assert_ne!(truncated, fraction);
    assert!(crate::IntType::I64.try_cast(fraction).is_err());
    let integer = Number::from(u64::MAX);
    let rounded = Number::from_f64(integer.to_f64().unwrap()).unwrap();
    assert_ne!(rounded, integer);
    assert_eq!(rounded.class(), Number::from(0f64).class());
}

#[test]
fn radix_parsing_delegates_and_rejects_unsupported_forms() {
    use num::traits::{Num, NumAssign};

    use crate::{Error, ErrorKind};

    fn decimal<T>(source: &str)
    where
        T: NumberInstance
            + PartialEq
            + Num<FromStrRadixErr = Error>
            + std::str::FromStr<Err = Error>,
    {
        match (T::from_str_radix(source, 10), source.parse::<T>()) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                assert!(a.class() == b.class());
            }
            (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind()),
            pair => panic!("decimal mismatch: {pair:?}"),
        }
    }
    fn add<T: NumAssign + Copy>(mut value: T) -> T {
        value += value;
        value
    }
    assert_eq!(add(Int::I8(3)), Int::I8(6));

    macro_rules! types {
        ($($ty:ty),+) => {$(
            for input in ["0", "1", "-1", "1.5", "true", "1+2i", "NaN", " 1 ", ""] { decimal::<$ty>(input); }
            for radix in [0, 1, 37, u32::MAX] {
                assert_eq!(<$ty>::from_str_radix("bad", radix).unwrap_err().kind(), ErrorKind::InvalidInput);
            }
        )+};
    }
    types!(Boolean, Int, UInt, Float, Complex, Number);
    assert_eq!(
        Int::from_str_radix("-8000000000000000", 16).unwrap(),
        Int::I64(i64::MIN)
    );
    assert_eq!(
        UInt::from_str_radix("ffffffffffffffff", 16).unwrap(),
        UInt::U64(u64::MAX)
    );
    assert_eq!(
        Number::from_str_radix("  ffffffffffffffff  ", 16).unwrap(),
        Number::from(u64::MAX)
    );
    assert_eq!(
        Number::from_str_radix("10", 2).unwrap().class(),
        Number::from(2i64).class()
    );
    assert_eq!(Int::from_str_radix("z", 36).unwrap(), Int::I64(35));
    for input in ["2", "-", "0b10"] {
        assert!(UInt::from_str_radix(input, 2).is_err());
    }
    assert!(Number::from_str_radix("10000000000000000", 16).is_err());
    for value in [u64::MAX as u128 + 1, u64::MAX as u128 + 2] {
        assert_eq!(
            Number::from_str_radix(&value.to_string(), 10)
                .unwrap_err()
                .kind(),
            ErrorKind::Overflow
        );
        assert!(Number::from_str_radix(&format!("{value:x}"), 16).is_err());
    }
    assert_eq!(
        Number::from_str_radix(&(i64::MIN as i128 - 1).to_string(), 10)
            .unwrap_err()
            .kind(),
        ErrorKind::Underflow
    );

    for radix in [2, 16, 36] {
        assert_eq!(
            Boolean::from_str_radix("true", radix).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            Float::from_str_radix("10", radix).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            Complex::from_str_radix("10", radix).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
    }
}

#[test]
fn negation_preserves_native_overflow_and_dynamic_mapping() {
    use std::hint::black_box;
    use std::panic::catch_unwind;

    use num::traits::{CheckedNeg, WrappingNeg};

    use crate::ErrorKind;

    macro_rules! signed {
        ($($ty:ty),+) => {$(
            for value in [<$ty>::MIN, -1, 0, 1, <$ty>::MAX] {
                let scalar = Int::from(value);
                let native = catch_unwind(|| -black_box(value));
                let ordinary = catch_unwind(|| -black_box(scalar));
                match (native, ordinary) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(actual, Int::from(expected));
                        assert_eq!(actual.class(), scalar.class());
                    }
                    (Err(_), Err(_)) => {},
                    _ => panic!("negation overflow mismatch"),
                }
                assert_eq!(CheckedNeg::checked_neg(&scalar), value.checked_neg().map(Int::from));
                assert_eq!(WrappingNeg::wrapping_neg(&scalar), Int::from(value.wrapping_neg()));
                if value == <$ty>::MIN { assert_eq!(scalar.try_neg().unwrap_err().kind(), ErrorKind::Overflow); }
                assert_eq!(scalar, Int::from(value));
            }
        )+};
    }
    signed!(i8, i16, i32, i64);

    macro_rules! unsigned {
        ($($ty:ty),+) => {$(
            for value in [0, 1, <$ty>::MAX] {
                let scalar = UInt::from(value);
                assert_eq!(CheckedNeg::checked_neg(&scalar), value.checked_neg().map(UInt::from));
                assert_eq!(WrappingNeg::wrapping_neg(&scalar), UInt::from(value.wrapping_neg()));
                if value != 0 { assert_eq!(scalar.try_neg().unwrap_err().kind(), ErrorKind::Underflow); }
            }
        )+};
    }
    unsigned!(u8, u16, u32, u64);
    for value in [false, true] {
        let number = Number::from(value);
        let expected = Number::from(-i16::from(value));
        assert_eq!(-number, expected);
        assert_eq!(number.try_neg().unwrap(), expected);
        assert_eq!(number.wrapping_neg(), expected);
        assert_eq!(number.try_neg().unwrap().class(), expected.class());
    }
    for (input, expected) in [
        (Number::from(255u8), Number::from(-255i16)),
        (Number::from(32768u16), Number::from(i16::MIN)),
        (Number::from(1u32 << 31), Number::from(i32::MIN)),
        (Number::from(1u64 << 63), Number::from(i64::MIN)),
    ] {
        assert_eq!(input.try_neg().unwrap(), expected);
        assert_eq!((-input).class(), expected.class());
    }
    for (input, wrapped) in [
        (Number::from(u16::MAX), Number::from(1i16)),
        (Number::from(u32::MAX), Number::from(1i32)),
        (Number::from(u64::MAX), Number::from(1i64)),
    ] {
        assert_eq!(input.try_neg().unwrap_err().kind(), ErrorKind::Underflow);
        assert!(catch_unwind(|| -black_box(input)).is_err());
        assert_eq!(input.wrapping_neg(), wrapped);
        assert_eq!(input.wrapping_neg().class(), wrapped.class());
    }
}

#[test]
fn reciprocal_and_nonfinite_unary_operations_keep_their_contracts() {
    use num::traits::Inv;

    use crate::{ErrorKind, NumberClass};

    macro_rules! real {
        ($($ty:ty),+) => {$(
            for value in [-0. as $ty, 0., 2., -2., <$ty>::MAX, <$ty>::from_bits(1), <$ty>::INFINITY, <$ty>::NAN] {
                let scalar = Float::from(value);
                let negated = -scalar;
                let reciprocal = Inv::inv(scalar);
                for (actual, expected) in [
                    (f64::from(negated), (-value) as f64),
                    (f64::from(reciprocal), (1. / value) as f64),
                ] {
                    if expected.is_nan() { assert!(actual.is_nan()); }
                    else { assert_eq!(actual.to_bits(), expected.to_bits()); }
                }
                assert_eq!(negated.class(), scalar.class());
                assert_eq!(reciprocal.class(), scalar.class());
                assert_eq!(scalar.try_neg().unwrap(), negated);
                match (scalar.try_recip(), scalar.class().one().try_div(scalar)) {
                    (Ok(a), Ok(b)) => { assert_eq!(a, b); assert_eq!(a.class(), scalar.class()); }
                    (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind()),
                    _ => panic!("checked reciprocal mismatch"),
                }
            }
            assert_eq!(Float::from(0 as $ty).try_recip().unwrap_err().kind(), ErrorKind::DivisionByZero);
            assert_eq!(Float::from(<$ty>::from_bits(1)).try_recip().unwrap_err().kind(), ErrorKind::Overflow);
        )+};
    }
    real!(f32, f64);
    for value in [
        Complex::from([-0f32, 0.]),
        Complex::from([2f64, -3.]),
        Complex::from([f64::INFINITY, -0.]),
        Complex::from([f64::NAN, 1.]),
    ] {
        let negated = -value;
        let original = crate::_Complex::<f64>::from(value);
        let actual = crate::_Complex::<f64>::from(negated);
        for (actual, expected) in [(actual.re, -original.re), (actual.im, -original.im)] {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert_eq!(value.try_neg().unwrap(), negated);
        assert_eq!(Inv::inv(value), value.class().one() / value);
        assert_eq!(value.recip().class(), value.class());
        match (value.try_recip(), value.class().one().try_div(value)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind()),
            _ => panic!("complex reciprocal mismatch"),
        }
    }
    let tiny_component = Complex::from([2f64, f64::from_bits(1)]);
    assert_eq!(
        tiny_component.try_recip().unwrap_err().kind(),
        ErrorKind::Underflow
    );
    let uncertified = Complex::from([1e300f64, 1e-300]);
    assert_eq!(
        uncertified.try_recip().unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        uncertified.try_recip().unwrap_err().kind(),
        uncertified
            .class()
            .one()
            .try_div(uncertified)
            .unwrap_err()
            .kind()
    );
    for (input, expected) in [
        (Number::from(2i8), Number::from(0.5f32)),
        (Number::from(2i64), Number::from(0.5f64)),
        (Number::from(2u64), Number::from(0.5f64)),
        (Number::from(true), Number::from(1f32)),
    ] {
        assert_eq!(input.recip(), expected);
        assert_eq!(input.recip().class(), expected.class());
        assert_eq!(input.try_recip().unwrap().class(), expected.class());
    }
    let zero = Number::from(false);
    assert_eq!(
        zero.try_recip().unwrap_err().kind(),
        ErrorKind::DivisionByZero
    );
    assert!(matches!(zero.recip(), Number::Float(v) if f64::from(v).is_infinite()));
}

#[test]
fn signedness_and_family_bounds_match_native_contracts() {
    use num::traits::{Bounded, Signed, Unsigned};

    use crate::{FloatType, UIntType};

    fn unsigned<T: Unsigned>(value: T) -> T {
        value
    }
    assert_eq!(unsigned(UInt::U64(9)), UInt::U64(9));

    macro_rules! floats {
        ($($ty:ty),+) => {$(
            for value in [-0. as $ty, 0., -1., 1., <$ty>::INFINITY, <$ty>::NEG_INFINITY, <$ty>::NAN] {
                let scalar = Float::from(value);
                assert_eq!(Signed::is_positive(&scalar), Signed::is_positive(&value));
                assert_eq!(Signed::is_negative(&scalar), Signed::is_negative(&value));
                let sign = Signed::signum(&scalar);
                let expected = Signed::signum(&value);
                assert_eq!(sign.class(), scalar.class());
                if expected.is_nan() { assert!(f64::from(sign).is_nan()); }
                else { assert_eq!(f64::from(sign).to_bits(), (expected as f64).to_bits()); }
                assert_eq!(Signed::abs(&scalar), NumberInstance::abs(scalar));
                let delta = Signed::abs_sub(&scalar, &Float::from(0 as $ty));
                let expected = Signed::abs_sub(&value, &(0 as $ty));
                if expected.is_nan() { assert!(f64::from(delta).is_nan()); }
                else { assert_eq!(f64::from(delta).to_bits(), (expected as f64).to_bits()); }
                assert_eq!(delta.class(), scalar.class());
            }
        )+};
    }
    floats!(f32, f64);
    let delta = Signed::abs_sub(&Float::F32(3.), &Float::F64(1.));
    assert_eq!(delta, Float::F64(2.));
    assert_eq!(delta.class(), FloatType::F64);
    assert_eq!(<Int as Bounded>::min_value(), Int::I64(i64::MIN));
    assert_eq!(<Int as Bounded>::min_value().class(), IntType::I64);
    assert_eq!(<Int as Bounded>::max_value(), Int::I64(i64::MAX));
    assert_eq!(<UInt as Bounded>::min_value().class(), UIntType::U64);
    assert_eq!(<UInt as Bounded>::max_value(), UInt::U64(u64::MAX));
    assert_eq!(<Float as Bounded>::min_value().class(), FloatType::F64);
    assert_eq!(<Float as Bounded>::min_value(), Float::F64(f64::MIN));
    assert_eq!(<Float as Bounded>::max_value(), Float::F64(f64::MAX));
    assert_eq!(<Boolean as Bounded>::min_value(), Boolean::from(false));
    assert_eq!(<Boolean as Bounded>::max_value(), Boolean::from(true));
    assert_eq!(
        <Int as num::traits::bounds::LowerBounded>::min_value(),
        <Int as Bounded>::min_value()
    );
}
