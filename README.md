# number-general
A generic Rust number type with support for basic math operations, (de)serialization, and casting with
[safecast](https://github.com/TinyChain-Inc/safecast).

Generic text parsing delegates to the existing signed, unsigned, floating-point,
complex, and Boolean parsers in that order. Signed integers retain precedence;
wide unsigned values, complex scientific notation, and pure imaginary forms are
accepted without a separate token-classification heuristic.

Example usage:
```rust
use number_general::{Int, Number};
use safecast::CastFrom;

let sequence: Vec<Number> = serde_json::from_str("[true, 2, 3.5, -4, [1.0, -0.5]]").unwrap();
let actual = sequence.into_iter().product();

assert_eq!(actual, Number::from(num::Complex::<f64>::new(-28., 14.)));
assert_eq!(Int::cast_from(actual), Int::from(-28));
```

Numeric equality and comparison preserve exact values across representations;
arithmetic retains its dtype-promotion rules. Signed zero compares equally and
NaNs use a canonical equality/hash representation. Standard hashes agree for
equal numbers. Primitive cryptographic encodings remain type-specific.

Complex magnitudes retain their component precision and use a stable norm.
A complex number is false exactly when both components are zero; NaNs are truthy.

Collation orders all real-valued numbers numerically before non-real complex
numbers. Non-real complex values use a double-precision norm; distinct values may
collate equally. JSON may spell an f32 with its full exactly represented value.
These corrections are unreleased ordering changes: recreate development stores
whose indexes relied on the earlier comparisons rather than reopening them under
new semantics.

## Numerical contracts

Choose the surface that fits the caller:

| Surface | Behavior |
| --- | --- |
| Ordinary operators and methods | Floating/complex IEEE behavior; integer overflow follows the application's Rust overflow settings |
| `try_*` methods | Structured errors for overflow, underflow, invalid domains, and unrepresentable results |
| `wrapping_*` methods | Explicit integer wrapping with ordinary promotion and floating/complex IEEE behavior |

Ordinary same-family integer operations promote widths before primitive Rust
arithmetic. Mixed signed/unsigned operations instead check their widened
mathematical result and panic if it cannot fit the selected destination, in every
profile. Assignments calculate the result before replacing their receiver.
TinyChain request execution uses checked operations; it does not rely on native
panics for arithmetic validation.

Promotion follows Boolean → unsigned → signed → float → complex, taking the
maximum width after family conversion. Existing unsigned-to-signed mappings are
retained, including U8→I16; Boolean→Int also uses I16. Mixed integer arithmetic
checks the result after evaluation, allowing cancellation even when an operand
alone cannot fit the destination.

### Checked operations

`NumberInstance` provides `try_add`, `try_sub`, `try_mul`, `try_div`, `try_rem`,
`try_abs`, `try_exp`, `try_ln`, `try_log`, `try_pow`, `try_round`, `try_into_type`,
`try_sum`, and `try_product`. `Trigonometry` has corresponding checked methods;
`NumberClass::try_cast` checks conversions. Custom implementations supply both
ordinary and checked numerical methods, without an additional policy trait.

Errors expose `Error::kind()`: `InvalidInput`, `Overflow`, `Underflow`,
`DivisionByZero`, `Domain`, `Unrepresentable`, or `Unsupported`.
`Error::with_kind` constructs a classified error; `Error::new` retains its existing
input-error diagnostic behavior.

Checked integer arithmetic uses widened intermediates and checks the selected
destination. Above the maximum is overflow; below the minimum is underflow.
Division truncates toward zero. Signed minimum divided by minus one errors; its
checked remainder is representable zero.

For finite floating and complex inputs, checked operations reject invalid domains,
new nonfinite results, and nonzero values lost to zero. Representable subnormals
and ordinary precision rounding are allowed. Cancellation, intentional rounding to
zero, and other defined zero results succeed. Checked real operations and retained
complex arithmetic preserve IEEE propagation from explicit NaN/infinity inputs.

Checked real transcendental functions, rounding and complex magnitude use the
pinned `libm` 0.2.16 software kernels, with
`force-soft-floats` overriding architecture/intrinsic features even when enabled
by another dependency. Ordinary and floating/complex wrapping methods retain
native Rust/backend math. No runtime backend selection or fallback is performed. Basic arithmetic uses
fixed-width Rust primitives under the IEEE environment described below.

For this implementation and a normal IEEE floating-point environment (nearest-even
arithmetic, subnormals enabled, no fast-math transformations), the checked contract
targets matching finite result bits, dtypes, signed zeros and error kinds across
validated builds/targets. NaN payloads are unspecified. This is not a promise for
arbitrary compiler/library versions or a modified floating-point environment.
Kernel/compiler upgrades must rerun the portability vectors and accuracy
references before extending the validated configuration. Custom numeric
implementations remain responsible for their own methods.
Conversions to `usize` are explicitly pointer-sized and outside that contract.

### Certified checked complex operations

| Operation | Supported inputs |
| --- | --- |
| Arithmetic, magnitude, rounding | Existing general complex support, including IEEE nonfinite arithmetic; exact cancellation checks are retained |
| Power | Bounded integral real exponents, including integral floats; complex and nonintegral exponents return `Unsupported` |
| Exponential | Finite real axis |
| Natural logarithm | Finite nonzero coordinate axes, preserving signed-zero branch sides |
| Logarithm | Those axes with a finite positive real base other than one |
| sin/cos/tan, sinh/cosh/tanh, asin/acos/atan, asinh/atanh | Existing finite real/imaginary-axis identities within their real-function domains; imaginary atan requires magnitude below one, and imaginary asinh requires magnitude at most one |
| acosh | Finite real axis at or above one |

The real domains for asin/acos are [-1, 1], for atanh (-1, 1), and for acosh
[1, infinity). Known zero/pole domain failures remain errors. Other complex
transcendental inputs, including nonfinite inputs, return `Unsupported` before
numerical evaluation. Complex-tagged logarithm bases and exponents are
unsupported even when their imaginary component is zero. Ordinary complex
methods remain available; checked methods
never retry through them. Auxiliary axis signs also use portable kernels.

Complex multiplication, division and remainder retain exact rational checks for
ambiguous zero/nonfinite components. That exceptional path allocates bounded
arbitrary-precision temporaries; general transcendental result guessing is removed.

Checked casts reject out-of-range values, lost fractions or imaginary components,
and Boolean destinations other than numeric zero or one. Finite precision rounding
is allowed; narrowing overflow and nonzero-to-zero underflow are errors.
`CastFrom`, `NumberClass::cast`, and `into_type` remain explicitly lossy conversions.
All four conversion surfaces resolve wildcard widths from the source.

### Powers, rounding, and aggregates

Integer-result powers retain the base dtype. Exponents must be finite integral
values with signed magnitude at most `u64::MAX`; fractional exponents are
`Unrepresentable`, and unsupported magnitudes are `Unsupported`. Negative powers
succeed only when the integer result is exact (bases ±1); zero to a negative power
errors. Dynamic `Number` retains integer evaluation for nonnegative integral exponents,
even when represented as floats, and promotes fractional or negative powers to
floats. Binary math chooses computation precision before converting its inputs;
a wider logarithm base no longer follows an intermediate f32 conversion.
Integer exponent inputs retain their full width, including for floating/complex
bases, so odd/even parity is not lost to floating conversion.

Floating `round` and `try_round` return floating values, preserving signed zero.
Use a checked cast when an integer is required. Boolean numerical functions use
0/1, while Boolean logical arithmetic remains unchanged.

Nonempty numeric aggregates start with their first element and consume inputs
left-to-right without zero shortcuts. Checked aggregates stop at the first error;
empty aggregates retain their existing identities. Concrete Boolean aggregates
retain logical short-circuiting.

### Explicit wrapping

`Int`, `UInt`, and `Number` provide inherent `wrapping_add`, `wrapping_sub`,
`wrapping_mul`, `wrapping_div`, `wrapping_rem`, `wrapping_abs`, `wrapping_pow`,
`wrapping_sum`, and `wrapping_product` methods. They return values directly,
without changing the ordinary or checked API or adding requirements to custom
numeric traits.

Wrapping changes integer overflow behavior, not promotion. Mixed widths and
signed/unsigned operands select the same result type as ordinary arithmetic.
Integer calculations wrap into that destination after evaluating the operands;
floating/complex results use ordinary IEEE/backend behavior. Boolean arithmetic
retains its logical rules. Explicit wrapping is independent of the application's
Rust overflow settings.

Integer division and remainder by zero panic, as with Rust's wrapping methods.
Signed minimum divided by minus one wraps to the minimum, and its remainder is
zero. Use `try_div`/`try_rem` for structured failures instead. Floating division by
zero retains IEEE behavior.

Typed integer `wrapping_pow` takes a `u64` exponent and retains the base dtype.
Dynamic `Number::wrapping_pow` takes `Number`: nonnegative integer powers wrap,
while negative or fractional powers follow ordinary promotion to floats. Complex
powers use the existing backend. `wrapping_sum` and `wrapping_product` seed from
the first value, consume left-to-right without zero shortcuts, and keep the
ordinary empty identities. No destination dtype argument is required.

For example, `Number::from(127i8).wrapping_add(Number::from(1u8))` is `128i16`,
while two I8 operands wrap to `-128i8`. Floating functions remain `sin`, `exp`,
`round`, and their checked counterparts; there is no parallel set of numerical
methods or mask-valued comparisons. Scalar/key equality and hashing remain
canonical, including NaNs.

ha-ndarray retains its own elementwise contract. Its integer zero divisors return
zero; scalar wrapping methods panic. Typed array powers retain their dtype for
negative exponents; dynamic scalar powers can promote. Array comparisons use
IEEE predicates, whereas scalar/key equality canonicalizes NaNs. Conformance
tests check shared behavior and assert these differences explicitly.

### Migration

- Checked real math uses pinned software kernels; final rounding can differ from
  native math. Portability vectors run in Linux and Windows CI; a local Linux pass
  does not by itself establish Windows agreement.
- Checked complex transcendentals now support the certified subset above and
  reject other inputs before evaluation.
- Integral floating exponents preserve integer bases and dtypes across ordinary,
  checked and wrapping powers. Logarithms prepare both arguments before conversion.

- The fixed-dtype scalar interface has been removed. Use ordinary numerical
  methods and casts, checked methods, or explicit wrapping arithmetic. Mixed
  numeric representations are supported through promotion.

- Replace implicit `UInt`→`Int`/`usize` conversions with `TryFrom` when loss is an
  error, or explicit `CastFrom` when the old lossy behavior is intended.
- Floating rounding now returns Float; add a checked integer cast if needed.
- Wildcard class casts now use source widths consistently with `into_type`;
  request a concrete target to fix a width.
- Boolean→Int uses I16, and Boolean trigonometry evaluates 0/1.
- Nonempty numeric aggregates preserve their first element's dtype until a later
  operand promotes it. Empty identities are unchanged.
- Custom numeric implementations must provide ordinary as well as checked
  methods. Generic logarithms no longer require `Float: From<N>`.

Equality, hashing, collation, codecs, and persisted encodings are unchanged by
these numerical API changes. ha-ndarray's production operations and kernels retain
their existing contract.

### WASM validation

Default and all-feature Rust libraries compile for `wasm32-unknown-unknown`.
The focused `--test wasm` integration target runs the shared checked portability
vectors and overflow classification through the wasm-bindgen test runner in Node.
The existing native portability test uses the same assertions. This verifies the
selected vectors rather than every operation on every WASM host. No production
JavaScript dependency or numerical policy is introduced by the tests.

## Generic numerical traits

The built-in enums implement these `num_traits` contracts, available through the
existing `num::traits` re-export:

- `ConstZero`, `ConstOne`, `Zero`, `One`, `CheckedAdd`, `CheckedSub`, `CheckedMul`, `CheckedDiv`, and
  `Pow<Number>` for `Boolean`, `Int`, `UInt`, `Float`, `Complex`, and `Number`.
- `WrappingAdd`, `WrappingSub`, `WrappingMul`, and `WrappingNeg` for `Int`, `UInt`, and `Number`.
- `ToPrimitive`, `FromPrimitive`, and `Num` for all six enums.
- `Neg` for `Int`, `Float`, `Complex`, and `Number`; `CheckedNeg` also for `UInt`.
- `Inv` for `Float`, `Complex`, and `Number`.
- `Signed` for `Float`, `Unsigned` for `UInt`, and `Bounded` for `Boolean`, `Int`, `UInt`, and `Float`.

Identities have one constant definition per enum; the function constructors return
those constants. Arithmetic adapters delegate to their owning methods. Checked adapters return
`None` for every error; use `try_*` when the error kind or diagnostic matters.
Power uses ordinary numerical behavior, including its existing panic conditions.
Wrapping retains promotion and does not require matching operand dtypes.

Parameterless identities use Bool for `Boolean` and `Number`, I16 for `Int`, U8
for `UInt`, F32 for `Float`, and C32 for `Complex`. Their predicates recognize
zero and one across widths, including signed zeros. `set_zero` and `set_one`
preserve the receiver's dtype; combining a constructed identity with another
value can promote that value under the ordinary rules. Importing overlapping
traits can make method syntax ambiguous; use fully qualified calls such as
`num::traits::Pow::pow(value, exponent)`. Ordinary power is owned by that trait;
`NumberInstance` inherits `Pow<Number, Output = Self>` and retains `try_pow`.

### Trait migration

`RealInstance` has been removed, including its unused sign helpers. Use
`ConstZero::ZERO`, `ConstOne::ONE`, and `Zero::is_zero` for identities. The former
`RealInstance` integer constants used I8; the canonical `Int` identities now use
I16, matching the existing default and identity constructors. The removed
zero-inclusive `is_positive` helper is not replaced with `Signed::is_positive`;
those contracts differ at zero, including floating signed zeros.

Custom `NumberInstance` implementations must implement
`num::traits::Pow<Number, Output = Self>` separately. Move the ordinary power
method body there and replace `<T as NumberInstance>::pow` with
`<T as num::traits::Pow<Number>>::pow`. Concrete callers import `Pow` to use
ordinary power method syntax; generic `T: NumberInstance` callers inherit it.
There is no forwarding alias. Checked powers, wrapping powers, and
receiver-dependent `NumberClass::zero/one` retain their existing owners.
`Trigonometry` and `FloatInstance` are also unchanged.

### Primitive conversion and radix contracts

Primitive conversion adapters follow the contained primitive's `num_traits`
conversion rules, including permitted truncation and floating precision loss.
All destination methods delegate directly, including 128-bit and pointer-sized
integers; integer conversion does not pass through a floating intermediate.
Complex exports require a zero imaginary component (either signed zero is
accepted); Boolean exports use 0/1. Non-real complex exports return `None`.
These adapters are not strict checked casts: cloud execution retains `try_cast`
and its structured diagnostics.

`FromPrimitive` chooses widths as follows:

| Destination | Result selection |
|---|---|
| `Boolean` | Accept exactly numeric zero or one, including negative floating zero. |
| `Int` | Preserve supported signed source widths; other sources use I64. |
| `UInt` | Preserve supported unsigned source widths; other sources use U64. |
| `Float` | Preserve f32/f64 sources; other sources use F64. |
| `Complex` | All primitive sources use C64 with zero imaginary part. |
| `Number` | Preserve supported primitive families and widths. i128 uses I64, then U64 for larger representable positive values; u128 uses U64. Larger integers return `None`, without a floating fallback. |

Pointer-sized imports use native pointer width. Cross-family range rejection and
float-to-integer truncation delegate to primitive conversions; no operand is
prematurely narrowed through another integer width.

Complete decimal integer literals remain integers: values above U64 return
`Overflow`, and negative values below I64 return `Underflow`. This includes
optional signs and leading zeros. Explicit decimal points, scientific notation,
and complex syntax retain their existing floating behavior. Plain `Display`
text does not encode dtype; use typed serialization when metadata must survive.

`TryFrom<UInt> for Int` uses the destination family's full I64 capacity and returns
I64 for every successful conversion, matching wide cross-family primitive
imports. Class-directed checked casts retain their explicit or source-inferred
widths, and explicit lossy casts retain their existing behavior. This intentionally
replaces rejection of representable U16/U32 values and changes successful narrow
conversion result dtypes to I64.

`Num::from_str_radix` uses the existing `FromStr` for radix 10, including its
whitespace and dynamic type-selection behavior. Other radices 2–36 support
`Int`, `UInt`, and integer-only `Number` parsing through i64/u64 parsers. Dynamic
parsing is signed-first and trims its input. Unsupported nondecimal Boolean,
Float and Complex requests return `Unsupported`. Invalid radices return
`InvalidInput` before primitive parsing. No prefix autodetection or new floating
parser is introduced. `NumAssign` follows the upstream blanket implementation.

### Negation and reciprocal

Ordinary unary `-` preserves Int/Float/Complex widths. Signed integer overflow
follows Rust settings; floating and complex components are negated directly,
retaining signed zeros and IEEE nonfinite behavior. `try_neg` returns `Overflow`
for signed minima. Typed unsigned `try_neg` accepts only zero and returns
`Underflow` otherwise; `UInt` has no ordinary `Neg` implementation.
`wrapping_neg` retains typed integer widths.

Dynamic `Number` Boolean/unsigned negation uses existing signed-family mappings:
Boolean, U8 and U16 select I16; U32 selects I32; U64 selects I64. Negation happens
in widened arithmetic before narrowing, so unsigned values mapping to a signed
minimum succeed. Larger negative results return `Underflow` from `try_neg`;
ordinary `-` panics in every profile. Dynamic wrapping negation wraps into the
selected signed dtype. Thus typed UInt wrapping negation stays unsigned, while
`Number` wrapping negation may select a signed family. Int variants retain their
existing ordinary/checked/wrapping overflow contracts.

`recip` and `Inv` delegate to ordinary division by a same-dtype one; `try_recip`
delegates to checked division. Float/Complex retain their dtype. Dynamic integer
and Boolean operands first use the existing floating-family mapping: 64-bit
integers select F64, narrower integers and Booleans select F32. Ordinary zero
reciprocals follow existing IEEE division; checked reciprocals report the existing
division, overflow and underflow errors. There is no typed integer reciprocal.
These checked unary methods are inherent and add no custom `NumberInstance`
requirements.

### Signs and family-wide bounds

`Signed<Float>` delegates sign classification and signum to the corresponding
primitive trait, preserving width. Positive/negative floating zeros and NaNs
follow upstream sign-bit behavior, not the removed zero-inclusive sign helpers.
Mixed-width `abs_sub` promotes within the floating family before primitive
execution. `Unsigned` is implemented only for UInt.

`Bounded` describes an entire enum family: Boolean false/true, I64 extrema, U64
zero/maximum, and finite F64 extrema. It does not report the bounds of an existing
I8 or F32 value. Lower/upper bounds follow blanket implementations; there are no
Complex or Number bounds and no per-dtype bounds API.

### Excluded and deferred traits

| Trait family | Status and reason |
|---|---|
| `Float`, `FloatCore` | Incompatible with canonical scalar equality: their IEEE contract expects NaN to compare unequal to itself. |
| `PrimInt` | Incompatible with varying-width integer representations and promotion; it requires fixed-width binary integers. |
| `CheckedRem` | Incompatible with direct delegation: signed minimum remainder by minus one returns `None` upstream, while `try_rem` returns representable zero. |
| `NumCast`, `AsPrimitive` | Not implemented. Target dtype inference and unconditional lossy casts remain separate from primitive adapters and explicit checked/lossy class casts. |
| `Signed<Int>` | Deferred: its documented minimum-absolute-value result differs from the crate's profile-sensitive ordinary `Int::abs`. |
| Signedness for Boolean, Complex or Number | Not implemented; the supported sign contract is confined to Float and UInt. |
| `FloatConst`, saturating and bitwise operations | Unsupported pending a consumer and an explicit precision/arithmetic contract. |

`NumOps`, `NumAssignOps`, `NumAssign`, and lower/upper bounds arise from upstream
blanket implementations where their requirements are satisfied; no manual
implementations are needed. Deferred capabilities are distinct from genuinely
incompatible contracts. Floating constants, saturation and bitwise operations do
not silently inherit another numerical policy.
