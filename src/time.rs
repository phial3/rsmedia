//! Time, and the exact rational numbers it is built on.
//!
//! Media metadata is rational by nature: a time base is `1 / sample_rate`, a
//! frame rate is `30_000 / 1_001` rather than `29.97`, a pixel aspect ratio is
//! `64 / 45`. This module holds [`Rational`] — the crate's exact rational — and
//! [`Time`], the timestamp type built on top of it.

use crate::error::{Result, RsmediaError};

use rsmpeg::avutil;
use rsmpeg::ffi;

use std::num::NonZeroI32;
use std::time::Duration;

/// An exact rational number `num / den`.
///
/// Media metadata is rational by nature: a time base is `1 / sample_rate`, a
/// frame rate is `30_000 / 1_001` rather than `29.97`, a pixel aspect ratio is
/// `64 / 45`. Storing those as `f32`/`f64` discards exactly the information that
/// makes them meaningful — no binary float is `30_000 / 1_001` — so this type
/// keeps the numerator and denominator as integers: the same representation, and
/// the same field width, as FFmpeg's `AVRational`.
///
/// This type is also the crate's **only** face for a rational. FFmpeg's
/// `AVRational` appears nowhere else in rsmedia: conversions in both directions
/// are [`From`]/[`Into`] here (see [`Rational::from`] and the `From<Rational>`
/// impl), so no other module — and no caller — ever has to name the raw type.
///
/// A [`Rational`] is normalised whenever it is built (see [`Rational::new`]):
///
/// * the denominator is positive,
/// * `num` and `den` are in lowest terms — a common divisor is divided out, so
///   `2 / 4` is stored as `1 / 2`,
/// * a zero numerator is stored as `0 / 1`.
///
/// Normalisation is what makes [`PartialEq`] mean "the same number" instead of
/// "the same spelling": `Rational::new(2, 4)? == Rational::new(1, 2)?`.
///
/// The fields are private so that every [`Rational`] this crate accepts or
/// returns is normalised; read them with [`Rational::num`] and [`Rational::den`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    num: i32,
    den: NonZeroI32,
}

impl Rational {
    /// The rational `0`, stored as `0/1`.
    ///
    /// Also what [`Rational::from`] yields for an FFmpeg rational that is
    /// "unset", which is why [`Time`] treats it as "no time base" rather than as
    /// a zero-width unit.
    pub const ZERO: Self = Self::integer(0);

    /// The rational `1`, i.e. `1/1`. The natural value for an *absent* scaling
    /// factor such as a pixel aspect ratio.
    pub const ONE: Self = Self::integer(1);

    /// The exact rational `num / den`, normalised as described on the type.
    ///
    /// Unlike a float entry point there is no approximation step: the value you
    /// pass is the value that is stored and that reaches FFmpeg.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::invalid_config`] when the number has no
    /// representation as a pair of `i32`:
    ///
    /// * `den == 0` — `x / 0` is not a number;
    /// * the **reduced** numerator or denominator still does not fit in `i32`.
    ///   Reduction happens *before* the range check, so this is narrower than it
    ///   looks: it needs `den == i32::MIN` (whose absolute value is `2^31`) with a
    ///   `num` that does not share enough factors to shrink it — `0 / i32::MIN`
    ///   and `i32::MIN / i32::MIN` both reduce into range and are accepted — or
    ///   `num == i32::MIN` with `den == -1`, whose normalised form is `2^31 / 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::Rational;
    ///
    /// // Film-on-NTSC, exactly. No float ever sees this value.
    /// let rate = Rational::new(24_000, 1_001)?;
    /// assert_eq!((rate.num(), rate.den()), (24_000, 1_001));
    ///
    /// // Normalised on construction: the same number, whichever way it is spelled.
    /// assert_eq!(Rational::new(50, 2)?, Rational::new(25, 1)?);
    /// assert_eq!(Rational::new(0, 7)?, Rational::integer(0));
    ///
    /// assert!(Rational::new(1, 0).is_err());
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub fn new(num: i32, den: i32) -> Result<Self> {
        if den == 0 {
            return Err(RsmediaError::invalid_config(format!(
                "a rational's denominator must not be zero (got {num}/0)"
            )));
        }

        // Everything below is computed in `i64`: `-i32::MIN` and `i32::MIN / -1`
        // both overflow `i32`, and the machine-word width lets the reduction
        // happen *before* the range check, so `i32::MIN / i32::MIN` (which is
        // really `1/1`) is accepted.
        let mut numerator = i64::from(num);
        let mut denominator = i64::from(den);
        if denominator < 0 {
            numerator = -numerator;
            denominator = -denominator;
        }

        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
        if divisor > 1 {
            let divisor = divisor as i64;
            numerator /= divisor;
            denominator /= divisor;
        }

        let out_of_range = || {
            RsmediaError::invalid_config(format!(
                "the rational {num}/{den} has no exact i32 representation"
            ))
        };
        let num = i32::try_from(numerator).map_err(|_| out_of_range())?;
        let den = i32::try_from(denominator)
            .ok()
            .and_then(NonZeroI32::new)
            .ok_or_else(out_of_range)?;

        Ok(Self { num, den })
    }

    /// The whole number `num` as a rational, i.e. `num / 1`.
    ///
    /// ```
    /// use rsmedia::Rational;
    ///
    /// let rate = Rational::integer(25);
    /// assert_eq!((rate.num(), rate.den()), (25, 1));
    /// assert_eq!(rate, Rational::new(25, 1)?);
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub const fn integer(num: i32) -> Self {
        Self {
            num,
            den: NonZeroI32::new(1).unwrap(),
        }
    }

    /// The unit fraction `1 / den` — the shape every time base has.
    ///
    /// This is the `const` counterpart of [`Self::new`] for the case that
    /// dominates media metadata: a time base is `1 / 1_000_000`, `1 /
    /// sample_rate` or `1 / fps`. Being `const` is what lets a time base be
    /// written as a `const` item, or used inside one, instead of being computed
    /// at every call site.
    ///
    /// `den` must be positive. That is not an arbitrary restriction: `1 / 0` is
    /// not a number, and `1 / -n` is simply `-1 / n`, which [`Self::new`] builds
    /// when that is what you meant. In a `const` context a bad `den` is a
    /// compile error; at run time it panics.
    ///
    /// ```
    /// use rsmedia::Rational;
    ///
    /// const MICROS: Rational = Rational::unit(1_000_000);
    /// assert_eq!((MICROS.num(), MICROS.den()), (1, 1_000_000));
    /// assert_eq!(MICROS, Rational::new(1, 1_000_000)?);
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub const fn unit(den: i32) -> Self {
        assert!(den > 0, "a unit fraction's denominator must be positive");
        Self {
            num: 1,
            den: NonZeroI32::new(den).unwrap(),
        }
    }

    /// The numerator. May be negative — a time base is not required to be positive.
    pub const fn num(&self) -> i32 {
        self.num
    }

    /// The denominator. Always positive and never zero.
    pub const fn den(&self) -> i32 {
        self.den.get()
    }

    /// The reciprocal, `den / num` — the frame rate implied by a time base, or
    /// the time base implied by a frame rate.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::invalid_config`] when the reciprocal has no
    /// `(i32, i32)` representation. The case that matters in practice is
    /// `num == 0`: zero has no reciprocal, its denominator would be zero, and
    /// [`Rational`] deliberately cannot represent that.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::Rational;
    ///
    /// // 25 fps and a 1/25 s time base are each other's reciprocal.
    /// let fps = Rational::integer(25);
    /// assert_eq!(fps.inverse()?, Rational::new(1, 25)?);
    /// assert_eq!(Rational::new(1, 25)?.inverse()?, fps);
    ///
    /// assert!(Rational::ZERO.inverse().is_err());
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub fn inverse(self) -> Result<Self> {
        Self::new(self.den.get(), self.num).map_err(|_| {
            RsmediaError::invalid_config(format!("the rational {self} cannot be inverted"))
        })
    }

    /// Whether this is exactly zero, i.e. [`Rational::ZERO`].
    ///
    /// A rational has exactly one zero spelling (`0/1`) — normalisation
    /// guarantees it — so this is the same test as `*self == Self::ZERO`,
    /// spelled out because "is the value zero" reads better than a comparison
    /// at call sites that branch on it.
    pub const fn is_zero(&self) -> bool {
        self.num == 0
    }

    /// The value as an `f64`.
    ///
    /// Lossy: this is the conversion this type exists to avoid. It is provided
    /// for printing, for thresholds, and for interop with float-based APIs; keep
    /// the [`Rational`] itself whenever exactness matters.
    pub fn as_f64(&self) -> f64 {
        f64::from(self.num) / f64::from(self.den.get())
    }
}

/// Euclid's algorithm on magnitudes. `gcd(0, x) == x`, which is what reduces a
/// zero numerator to the canonical `0 / 1`.
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

impl From<i32> for Rational {
    /// A whole number is the rational `num / 1`.
    fn from(num: i32) -> Self {
        Self::integer(num)
    }
}

impl From<Rational> for ffi::AVRational {
    /// Convert to FFmpeg's representation.
    ///
    /// Lossless in both directions of the value range: [`Rational::num`] and
    /// [`Rational::den`] are already `i32`, and the denominator is positive — the
    /// convention every FFmpeg API that consumes a rational expects.
    fn from(rational: Rational) -> Self {
        avutil::ra(rational.num, rational.den.get())
    }
}

impl From<ffi::AVRational> for Rational {
    /// Reads FFmpeg's own representation.
    ///
    /// FFmpeg writes `x/0` into a structure whose rational is *not known yet*:
    /// `AVFrame.time_base` before a decoder fills it in, `AVStream.avg_frame_rate`
    /// on a stream that has no rate, `AVCodecContext.time_base` before the codec
    /// is opened. That spelling is not a number, and [`Rational`] cannot hold it
    /// by construction — so it folds to [`Rational::ZERO`], the spelling every
    /// consumer in this crate already treats as "unset". A value outside the
    /// `(i32, i32)` range is only reachable through such a denominator and folds
    /// the same way.
    ///
    /// This is total on purpose: reading a rational out of an FFmpeg structure
    /// should not force every caller to invent a policy for a placeholder FFmpeg
    /// itself produces routinely. Use [`Rational::new`] when a zero denominator
    /// must be an error instead.
    fn from(rational: ffi::AVRational) -> Self {
        Self::new(rational.num, rational.den).unwrap_or(Self::ZERO)
    }
}

impl std::fmt::Display for Rational {
    /// Formats as `num/den`, the way FFmpeg and `ffprobe` print a rational:
    /// `30000/1001`, not `29.97`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.num, self.den.get())
    }
}

/// Represents a time or duration.
///
/// [`Time`] may represent a PTS (presentation timestamp), DTS (decoder timestamp) or a duration,
/// depending on the function that returns it.
///
/// [`Time`] may represent a non-existing time, in which case [`Time::has_value`] will return
/// `false`, and conversions to seconds will return `0.0`. FFmpeg's `AV_NOPTS_VALUE`
/// sentinel means exactly that, so it is folded into "no value" at the FFI
/// boundary (see [`Time::new`]) and by every accessor below.
///
/// Equality and ordering are expressed in **seconds**, not in `(time, time_base)`
/// pairs: two stamps that denote the same instant are equal even if their time
/// bases differ (see [`PartialEq`]).
///
/// A [`Time`] object may be aligned with another [`Time`] object, which produces an [`Aligned`]
/// object, on which arithmetic operations can be performed.
#[derive(Debug, Copy, Clone)]
pub struct Time {
    pub time: Option<i64>,
    /// The time base, as a [`Rational`] — see that type for why.
    ///
    /// [`Rational::ZERO`] is the "no time base" spelling: FFmpeg's own `0/0`
    /// placeholder is normalised to it on the way in, and every accessor below
    /// treats a zero time base as "nothing to convert".
    pub time_base: Rational,
}

impl Time {
    /// Create a new time by its time value and time base in which the time is expressed.
    ///
    /// `AV_NOPTS_VALUE` (FFmpeg's "no timestamp" sentinel, which streams and
    /// frames routinely carry) is normalised to `None` here — this is the FFI
    /// boundary, so callers can trust [`Time::has_value`] afterwards instead of
    /// re-checking the sentinel themselves.
    ///
    /// # Arguments
    ///
    /// * `time` - Relative time in `time_base` units.
    /// * `time_base` - Time base of source.
    pub fn new(time: Option<i64>, time_base: Rational) -> Time {
        Self {
            time: time.filter(|time| *time != ffi::AV_NOPTS_VALUE),
            time_base,
        }
    }

    /// Creates a new timestamp that represents one `nth` of a second — the
    /// instant `1 / nth` s, spelled as the tick `1` in a `1 / nth` time base
    /// rather than as a rounded float.
    ///
    /// # Arguments
    ///
    /// * `nth` - Denominator of the time in seconds as in `1 / nth`. [`i32`]
    ///   because that is the width of `AVRational`'s fields: the value the caller
    ///   writes is the value FFmpeg reads, with no narrowing in between.
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `nth == 0`: `1 / 0` is not a
    /// number, and [`Rational`] cannot hold it. That is [`Rational::new`]'s own
    /// rule, not an extra one — it is the only case this constructor has to
    /// reject.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::Time;
    ///
    /// assert_eq!(Time::from_nth_of_a_second(4)?.as_secs_f64(), 0.25);
    /// assert!(Time::from_nth_of_a_second(0).is_err());
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub fn from_nth_of_a_second(nth: i32) -> Result<Self> {
        Ok(Self {
            time: Some(1),
            time_base: Rational::new(1, nth)?,
        })
    }

    /// Creates a new timestamp from a number of seconds.
    ///
    /// # Arguments
    ///
    /// * `secs` - Number of seconds.
    pub fn from_secs(secs: f32) -> Self {
        Self {
            time: Some((secs * TIME_BASE.den() as f32).round() as i64),
            time_base: TIME_BASE,
        }
    }

    /// Creates a new timestamp from a number of seconds.
    ///
    /// # Arguments
    ///
    /// * `secs` - Number of seconds.
    pub fn from_secs_f64(secs: f64) -> Self {
        Self {
            time: Some((secs * TIME_BASE.den() as f64).round() as i64),
            time_base: TIME_BASE,
        }
    }

    /// Creates a new timestamp with `time` time units, each represents one / `base_den` seconds.
    ///
    /// # Arguments
    ///
    /// * `time` - Relative time in `time_base` units. [`i64`] because that is
    ///   FFmpeg's own timestamp width (`AVFrame.pts`, `AVPacket.pts`).
    /// * `base_den` - Time base denominator i.e. time base is `1 / base_den`.
    ///   [`i32`], like `AVRational`'s fields.
    ///
    /// Both are the widths FFmpeg reads back, so neither is narrowed on the way
    /// in — see [`Self::from_nth_of_a_second`].
    ///
    /// # Errors
    ///
    /// Returns [`RsmediaError::InvalidConfig`] when `base_den == 0`, for the same
    /// reason [`Self::from_nth_of_a_second`] does: `1 / 0` is not a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmedia::Time;
    ///
    /// assert_eq!(Time::from_units(3, 2)?.as_secs_f64(), 1.5);
    /// assert!(Time::from_units(1, 0).is_err());
    /// # Ok::<(), rsmedia::RsmediaError>(())
    /// ```
    pub fn from_units(time: i64, base_den: i32) -> Result<Self> {
        Ok(Self {
            time: Some(time),
            time_base: Rational::new(1, base_den)?,
        })
    }

    /// Create a new zero-valued timestamp.
    ///
    /// 时间基取 [`TIME_BASE`]，因此与 `Time::new(Some(0), TIME_BASE)`（以及
    /// `Time::from_secs(0.0)`）完全相等 —— 早先这里硬编码 `1/90000`，会出现
    /// "同为 0 秒却不相等" 的荒谬结果。
    pub fn zero() -> Self {
        Time {
            time: Some(0),
            time_base: TIME_BASE,
        }
    }

    /// Whether the [`Time`] carries a usable value at all.
    ///
    /// Three "no time" spellings report `false`: no value whatsoever (`None`); the
    /// `AV_NOPTS_VALUE` sentinel FFmpeg writes when a stream simply has no
    /// timestamp; and a **zero time base** ([`Rational::ZERO`], what FFmpeg's own
    /// `0/0` folds to), which leaves a raw value nobody can interpret —
    /// [`Self::as_secs_f64`], [`std::fmt::Display`] and the comparisons all treat
    /// it as "nothing", so this predicate has to as well.
    ///
    /// This is the predicate to branch on before converting to seconds — converting
    /// a NOPTS would yield ≈ -9.2e12 seconds.
    /// [`Self::into_value`], the seconds conversions and the comparisons all agree
    /// with it (every one of them funnels through the same private helper).
    pub fn has_value(&self) -> bool {
        self.instant().is_some()
    }

    /// The raw value, with the `AV_NOPTS_VALUE` sentinel filtered out.
    ///
    /// [`Time::new`] already normalises the sentinel, but `time` is a public
    /// field, so a hand-built `Time { time: Some(AV_NOPTS_VALUE), .. }` is still
    /// possible; every accessor funnels through here so that there is only one
    /// notion of "has a value".
    fn value(&self) -> Option<i64> {
        self.time.filter(|time| *time != ffi::AV_NOPTS_VALUE)
    }

    /// The instant as an exact rational `(time, num, den)`, meaning
    /// `time * num / den` seconds — or `None` when there is nothing to compare:
    /// no value at all, or a zero time base.
    ///
    /// This is the key the comparison impls use (see [`compare_instants`]); the
    /// seconds conversions and [`std::fmt::Display`] go through
    /// [`Self::seconds_or_none`] instead, because printing wants a rounded
    /// number.
    ///
    /// The old "degenerate `0/0`" case is gone as a *separate* case: a zero
    /// denominator cannot exist in a [`Rational`], so the only degenerate time
    /// base left is [`Rational::ZERO`], and one test covers it.
    fn instant(&self) -> Option<(i64, i32, i32)> {
        let time = self.value()?;
        if self.time_base.is_zero() {
            return None;
        }
        Some((time, self.time_base.num(), self.time_base.den()))
    }

    /// The instant in seconds, or `None` when there is nothing to convert: no
    /// value at all, or a zero time base.
    ///
    /// Rounded to the nearest `f64`, so it is what [`Self::as_secs_f64`] and
    /// [`std::fmt::Display`] want and **not** what the comparisons use — those
    /// are exact (see [`Self::instant`]).
    ///
    /// The old "degenerate `0/0`" case is gone as a *separate* case: a zero
    /// denominator cannot exist in a [`Rational`], so the only degenerate time
    /// base left is [`Rational::ZERO`], and one test covers it.
    fn seconds_or_none(&self) -> Option<f64> {
        let time = self.value()?;
        if self.time_base.is_zero() {
            return None;
        }
        Some(time as f64 * self.time_base.as_f64())
    }

    /// Align the timestamp with another timestamp, which will convert the `rhs` timestamp to the
    /// same time base, such that operations can be performed upon the aligned timestamps.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side timestamp.
    ///
    /// # Return value
    ///
    /// Two timestamps that are aligned.
    pub fn aligned_with(&self, rhs: Time) -> Aligned {
        Aligned {
            lhs: self.value(),
            rhs: rhs
                .value()
                .map(|rhs_time| rhs_time.rescale(rhs.time_base, self.time_base)),
            time_base: self.time_base,
        }
    }

    /// Get number of seconds as floating point value.
    ///
    /// Single-precision on purpose (the historical API); it is computed from
    /// [`Self::as_secs_f64`] solely so the two can never disagree. Use the `f64`
    /// variant when precision matters.
    pub fn as_secs(&self) -> f32 {
        self.as_secs_f64() as f32
    }

    /// Get number of seconds as floating point value.
    ///
    /// Returns `0.0` when there is no usable value ([`Self::has_value`] is
    /// `false`, which covers a zero time base too), which also keeps the result
    /// finite — a NOPTS would otherwise turn into ≈ -9.2e12 seconds.
    pub fn as_secs_f64(&self) -> f64 {
        self.seconds_or_none().unwrap_or(0.0)
    }

    /// Convert to underlying time to `i64` (the number of time units).
    ///
    /// Returns `None` when there is no usable value — including the
    /// `AV_NOPTS_VALUE` sentinel **and a zero time base** (a raw count nobody can
    /// interpret) — so it agrees with [`Self::has_value`].
    ///
    /// Assumes that the caller knows the time base and applies it correctly when doing arithmetic
    /// operations on the time value.
    pub fn into_value(self) -> Option<i64> {
        self.instant().map(|(time, _, _)| time)
    }

    /// Align the timestamp along another `time_base`.
    ///
    /// # Arguments
    ///
    /// * `time_base` - Target time base.
    pub fn aligned_with_rational(&self, time_base: Rational) -> Time {
        Time {
            time: self
                .value()
                .map(|time| time.rescale(self.time_base, time_base)),
            time_base,
        }
    }
}

/////////////////////////////////
/////////////////////////////////

/// The microsecond time base (`1/1_000_000`), FFmpeg's `AV_TIME_BASE_Q` — the
/// unit [`Time`] uses for its "by seconds" constructors and for timestamps that
/// have no stream time base of their own.
pub const TIME_BASE: Rational = Rational::unit(1_000_000);

/// Rescale a timestamp between two time bases.
///
/// Implemented for every integer type that converts into `i64`
/// (`impl<T: Into<i64> + Clone>`), so a pts value can be rescaled in place:
/// `pts.rescale(from, to)`. That covers `i8`…`i64` and `u8`…`u32`, but **not**
/// `u64` / `usize` / `u128` / `i128` — cast those to `i64` first. Both time
/// bases are [`Rational`], like every other rational in this crate.
pub trait Rescale {
    fn rescale<S, D>(&self, source: S, destination: D) -> i64
    where
        S: Into<Rational>,
        D: Into<Rational>;

    fn rescale_with<S, D>(&self, source: S, destination: D, rounding: ffi::AVRounding) -> i64
    where
        S: Into<Rational>,
        D: Into<Rational>;
}

impl<T: Into<i64> + Clone> Rescale for T {
    fn rescale<S, D>(&self, source: S, destination: D) -> i64
    where
        S: Into<Rational>,
        D: Into<Rational>,
    {
        avutil::av_rescale_q(
            self.clone().into(),
            source.into().into(),
            destination.into().into(),
        )
    }

    fn rescale_with<S, D>(&self, source: S, destination: D, rounding: ffi::AVRounding) -> i64
    where
        S: Into<Rational>,
        D: Into<Rational>,
    {
        avutil::av_rescale_q_rnd(
            self.clone().into(),
            source.into().into(),
            destination.into().into(),
            rounding as _,
        )
    }
}

/// 两个时刻的**精确**比较：`time * num / den` 是有理数，用 `i128` 交叉相乘比大小，
/// 而不是先换算成 `f64`。
///
/// 浮点化过不了 [`Eq`] 的传递性：`9/15` 与 `3/5` 数学上相等，但两条不同的
/// 计算路径各带一次舍入就可能相差一个 ulp，于是 `a == b`、`b == c` 而 `a != c`
/// —— 放进 `HashSet`/排序里就是不稳定结果。[`Rational`] 保证 `den > 0`，故交叉
/// 相乘不会翻转符号。
///
/// `i128` 装得下最坏情况：`i64::MAX * i32::MAX * i32::MAX ≈ 4.3e37 < i128::MAX ≈ 1.7e38`。
///
/// "无值"（`None` 或零时间基）排在**最前面**且彼此相等，与 [`Time::has_value`]
/// 的语义一致。
fn compare_instants(lhs: &Time, rhs: &Time) -> std::cmp::Ordering {
    match (lhs.instant(), rhs.instant()) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (Some((lhs_time, lhs_num, lhs_den)), Some((rhs_time, rhs_num, rhs_den))) => {
            let lhs = i128::from(lhs_time) * i128::from(lhs_num) * i128::from(rhs_den);
            let rhs = i128::from(rhs_time) * i128::from(rhs_num) * i128::from(lhs_den);
            lhs.cmp(&rhs)
        }
    }
}

impl PartialEq for Time {
    /// Compares the instants, **in seconds**, not the `(time, time_base)` pairs:
    /// `Time::from_units(1, 4)` and `Time::new(Some(2), Rational::new(1, 2).unwrap())`
    /// both mean 0.5 s and are therefore equal.
    ///
    /// Requiring the raw fields to match used to make "the same instant" compare
    /// unequal whenever the time bases differed. Two "no value" times are equal
    /// (`None == None`); a "no value" time never equals a valued one. The result
    /// matches [`PartialOrd`]: `a == b` ⟺ `a.partial_cmp(&b) == Some(Equal)`.
    ///
    /// The comparison is exact — the two instants are cross-multiplied in `i128`
    /// instead of being converted to `f64` first, so unreduced time bases such as
    /// `9/15` and `3/5` compare equal even when the two floating-point paths
    /// differ by an ulp.
    fn eq(&self, other: &Self) -> bool {
        compare_instants(self, other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Time {}

impl PartialOrd for Time {
    /// Orders by seconds, "no value" first; [`PartialEq`] uses the same key, so
    /// the `PartialOrd` contract holds. Always `Some`, since the key is always
    /// comparable — and total, being an exact rational comparison rather than a
    /// floating-point one.
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(compare_instants(self, other))
    }
}

impl From<Duration> for Time {
    /// Convert from a [`Duration`] to [`Time`].
    #[inline]
    fn from(duration: Duration) -> Self {
        Time::from_secs_f64(duration.as_secs_f64())
    }
}

impl From<Time> for Duration {
    /// Convert from a [`Time`] to a Rust-native [`Duration`].
    fn from(timestamp: Time) -> Self {
        Duration::from_secs_f64(timestamp.as_secs_f64().max(0.0))
    }
}

impl std::fmt::Display for Time {
    /// Format [`Time`] as follows:
    ///
    /// * If the inner value is usable: the number of seconds, e.g. `0.5 secs`.
    /// * Otherwise: `none`.
    ///
    /// Printing the seconds (from [`Time::as_secs_f64`]) rather than the raw
    /// `time * time_base.num` numerator keeps `Display` panic-free: that product
    /// overflows `i64` for large timestamps and would panic in debug builds.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.seconds_or_none() {
            Some(secs) => write!(f, "{secs} secs"),
            None => write!(f, "none"),
        }
    }
}

/// This is a virtual object that represents two aligned times.
///
/// On this object, arthmetic operations can be performed that operate on the two contained times.
/// This virtual object ensures that the interface to these operations is safe.
#[derive(Debug, Clone)]
pub struct Aligned {
    lhs: Option<i64>,
    rhs: Option<i64>,
    time_base: Rational,
}

impl Aligned {
    /// Add two timestamps together.
    pub fn add(self) -> Time {
        self.apply(|lhs, rhs| lhs + rhs)
    }

    /// Subtract the right-hand side timestamp from the left-hand side timestamp.
    pub fn subtract(self) -> Time {
        self.apply(|lhs, rhs| lhs - rhs)
    }

    /// Apply operation `f` on aligned timestamps.
    ///
    /// The closure operates on the numerator of two aligned times.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to apply on the two aligned time numerator values.
    fn apply<F>(self, f: F) -> Time
    where
        F: FnOnce(i64, i64) -> i64,
    {
        match (self.lhs, self.rhs) {
            (Some(lhs_time), Some(rhs_time)) => Time {
                time: Some(f(lhs_time, rhs_time)),
                time_base: self.time_base,
            },
            _ => Time {
                time: None,
                time_base: self.time_base,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `num / den` for the tests. Every literal below is a valid rational except
    /// `rat(0, 0)`, which stands for "no time base" and folds to
    /// [`Rational::ZERO`] — exactly what FFmpeg's own `0/0` placeholder becomes
    /// on the way in.
    fn rat(num: i32, den: i32) -> Rational {
        Rational::new(num, den).unwrap_or(Rational::ZERO)
    }

    // ---- Rational ----

    #[test]
    fn test_rational_new_normalises_sign_and_terms() {
        // 分母为负 ⇒ 整体取反，分母恒正
        let rate = Rational::new(1, -2).unwrap();
        assert_eq!((rate.num(), rate.den()), (-1, 2));

        // 约分到最简
        assert_eq!(Rational::new(50, 2).unwrap(), Rational::new(25, 1).unwrap());
        assert_eq!(Rational::new(24_000, 1_001).unwrap().num(), 24_000);

        // 分子为 0 ⇒ 规范化成 0/1，与分母无关
        assert_eq!(Rational::new(0, 7).unwrap(), Rational::integer(0));
        assert_eq!(Rational::new(0, -7).unwrap(), Rational::integer(0));
    }

    #[test]
    fn test_rational_new_rejects_zero_denominator() {
        let err = Rational::new(1, 0).unwrap_err();
        assert!(err.is_invalid_config(), "got: {err}");
    }

    /// 规范化在范围检查**之前**做，所以 `i32::MIN / i32::MIN`（其实是 `1/1`）
    /// 必须被接受，而不是先按 `i32::MIN` 判成越界。
    #[test]
    fn test_rational_new_reduces_before_range_check() {
        let one = Rational::new(i32::MIN, i32::MIN).unwrap();
        assert_eq!(one, Rational::integer(1));

        // `i32::MIN` 与 `i32::MAX` 互素，约不掉，但两个分量都在范围内
        let almost_minus_one = Rational::new(i32::MIN, i32::MAX).unwrap();
        assert_eq!(
            (almost_minus_one.num(), almost_minus_one.den()),
            (i32::MIN, i32::MAX)
        );
        assert!(almost_minus_one.as_f64() < 0.0);
    }

    /// 只有两种真正无法用 `(i32, i32)` 表示的情况：`den == i32::MIN`，以及
    /// `(i32::MIN, -1)`（规范形是 `2^31 / 1`）。
    #[test]
    fn test_rational_new_rejects_unrepresentable_values() {
        for (num, den) in [(1, i32::MIN), (i32::MIN, -1)] {
            let err = Rational::new(num, den).unwrap_err();
            assert!(err.is_invalid_config(), "{num}/{den} gave: {err}");
        }
    }

    #[test]
    fn test_rational_integer_and_conversions() {
        assert_eq!(Rational::integer(25).as_f64(), 25.0);
        assert_eq!(Rational::from(25), Rational::integer(25));

        let rational = Rational::new(30_000, 1_001).unwrap();
        let ffi_rational: ffi::AVRational = rational.into();
        assert_eq!((ffi_rational.num, ffi_rational.den), (30_000, 1_001));
        assert_eq!(Rational::from(ffi_rational), rational);

        // 整数值往返：`Rational` 的字段宽度与 `AVRational` 一致，0/1 与 1/1 都能原样回来
        for value in [Rational::ZERO, Rational::ONE] {
            let raw: ffi::AVRational = value.into();
            assert_eq!(Rational::from(raw), value);
        }
    }

    /// FFmpeg 侧用 `x/0` 表示"尚未填写"；`Rational` 表示不了，读取时归一成 `ZERO`。
    /// 这是刻意的**全函数**读取口：调用方不该为 FFmpeg 自己产出的占位值准备策略。
    #[test]
    fn test_rational_from_ffi_folds_zero_denominator_to_zero() {
        for raw in [
            ffi::AVRational { num: 1, den: 0 },
            ffi::AVRational { num: 0, den: 0 },
            ffi::AVRational { num: -1, den: 0 },
        ] {
            assert_eq!(
                Rational::from(raw),
                Rational::ZERO,
                "{}/{} 应归一成 0/1",
                raw.num,
                raw.den
            );
        }

        // 严格的构造口仍然拒绝 `x/0`
        assert!(Rational::new(1, 0).unwrap_err().is_invalid_config());
    }

    #[test]
    fn test_rational_inverse() {
        let fps = Rational::integer(25);
        assert_eq!(fps.inverse().unwrap(), Rational::new(1, 25).unwrap());
        assert_eq!(
            Rational::new(1, 25).unwrap().inverse().unwrap(),
            fps,
            "互逆"
        );
        assert_eq!(Rational::integer(1).inverse().unwrap(), Rational::ONE);

        // 负分子：结果仍规范化为"分母为正"
        let negative = Rational::new(-3, 2).unwrap().inverse().unwrap();
        assert_eq!((negative.num(), negative.den()), (-2, 3));

        // 0 没有倒数（分母会是 0，`Rational` 表示不了）
        let err = Rational::ZERO.inverse().unwrap_err();
        assert!(err.is_invalid_config(), "got: {err}");
    }

    #[test]
    fn test_rational_is_zero() {
        assert!(Rational::ZERO.is_zero());
        assert!(Rational::new(0, 7).unwrap().is_zero());
        assert!(!Rational::ONE.is_zero());
        assert!(!Rational::new(-1, 2).unwrap().is_zero());
    }

    #[test]
    fn test_rational_display_prints_the_exact_rational() {
        assert_eq!(
            Rational::new(30_000, 1_001).unwrap().to_string(),
            "30000/1001"
        );
        assert_eq!(Rational::integer(25).to_string(), "25/1");
    }

    // ---- Time ----

    #[test]
    fn test_new() {
        let time = Time::new(Some(2), rat(3, 9));
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        assert_eq!(time.into_value(), Some(2));
    }

    #[test]
    fn test_aligned_with_rational() {
        let time = Time::new(Some(2), rat(3, 9));
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        let time = time.aligned_with_rational(rat(1, 9));
        assert_eq!(time.as_secs(), 2.0 / 3.0);
        assert_eq!(time.into_value(), Some(6));
    }

    #[test]
    fn test_from_nth_of_a_second() -> Result<()> {
        let time = Time::from_nth_of_a_second(4)?;
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 0.25);
        assert_eq!(time.as_secs_f64(), 0.25);
        assert_eq!(Duration::from(time), Duration::from_millis(250));
        Ok(())
    }

    /// `1 / 0` is refused rather than folded into a "no time base" [`Time`].
    ///
    /// Everything else is now out of reach instead of checked: the denominator is
    /// [`i32`] — `AVRational`'s own width — so there is no narrowing left for a
    /// value to wrap in. `from_nth_of_a_second(2^31 + 5)` used to become `1 / 5`
    /// at the `as i32` cast; it now does not compile.
    #[test]
    fn test_from_nth_of_a_second_rejects_a_zero_denominator() {
        let zero = Time::from_nth_of_a_second(0).unwrap_err();
        assert!(zero.is_invalid_config(), "{zero}");

        // The whole i32 range is usable, and reaches FFmpeg unchanged.
        let extreme =
            Time::from_nth_of_a_second(i32::MAX).expect("i32::MAX is a valid denominator");
        assert_eq!(extreme.time_base, Rational::new(1, i32::MAX).unwrap());
        assert_eq!(extreme.time_base.den(), i32::MAX);
    }

    #[test]
    fn test_from_secs() {
        let time = Time::from_secs(2.5);
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 2.5);
        assert_eq!(time.as_secs_f64(), 2.5);
        assert_eq!(Duration::from(time), Duration::from_millis(2500));
    }

    #[test]
    fn test_from_secs_f64() {
        let time = Time::from_secs(4.0);
        assert!(time.has_value());
        assert_eq!(time.as_secs_f64(), 4.0);
    }

    #[test]
    fn test_from_units() -> Result<()> {
        let time = Time::from_units(3, 5)?;
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 3.0 / 5.0);
        assert_eq!(Duration::from(time), Duration::from_millis(600));
        Ok(())
    }

    /// Both arguments are FFmpeg's own widths (`int64_t` ticks, `int` denominator),
    /// so the only case left to reject is `1 / 0`. Ticks used to be `usize`, which
    /// wraps to `-1` past `i64::MAX` on a 64-bit target; they are `i64` now.
    #[test]
    fn test_from_units_rejects_a_zero_denominator() -> Result<()> {
        assert!(Time::from_units(1, 0).unwrap_err().is_invalid_config());

        // The whole range is usable and reaches FFmpeg unchanged — including
        // negative ticks, which FFmpeg legitimately carries.
        let extreme = Time::from_units(i64::MAX, i32::MAX).expect("in-range arguments");
        assert_eq!(extreme.into_value(), Some(i64::MAX));
        let negative = Time::from_units(-3, 2)?;
        assert_eq!(negative.as_secs_f64(), -1.5);
        Ok(())
    }

    #[test]
    fn test_zero() {
        let time = Time::zero();
        assert!(time.has_value());
        assert_eq!(time.as_secs(), 0.0);
        assert_eq!(time.as_secs_f64(), 0.0);
        assert_eq!(Duration::from(time), Duration::ZERO);
        let time = Time::zero();
        assert_eq!(time.into_value(), Some(0));
    }

    /// `Time::zero()` 与同样表示 0 秒的值相等（无论时间基是否相同）。
    #[test]
    fn test_zero_equals_other_zero_values() {
        assert_eq!(Time::zero(), Time::new(Some(0), TIME_BASE));
        assert_eq!(Time::zero(), Time::from_secs(0.0));
        assert_eq!(Time::zero(), Time::from_secs_f64(0.0));
    }

    /// "无值" 与 `AV_NOPTS_VALUE` 都表示"没有可用时间戳"：`new` 在 FFI 边界把
    /// 哨兵归一为 `None`，所有取值口（`has_value` / `into_value` / 秒换算）一致。
    #[test]
    fn test_has_value_rejects_missing_and_nopts() {
        let missing = Time::new(None, TIME_BASE);
        let nopts = Time::new(Some(ffi::AV_NOPTS_VALUE), TIME_BASE);

        assert!(!missing.has_value());
        assert!(!nopts.has_value());
        assert_eq!(nopts.into_value(), None);
        assert_eq!(nopts.as_secs_f64(), 0.0);
        assert_eq!(nopts.as_secs(), 0.0);
        assert_eq!(nopts.to_string(), "none");
        assert_eq!(nopts, missing);

        assert!(Time::new(Some(0), TIME_BASE).has_value());
    }

    /// `partial_cmp` 与 `PartialEq` 同键（秒）：同一时刻即使时间基不同也相等且有序。
    #[test]
    fn test_partial_cmp_is_consistent_with_eq() {
        let a = Time::new(None, rat(1, 2));
        let b = Time::new(None, rat(1, 4));
        assert_eq!(a, b, "两个无值的时间戳相等");
        assert_eq!(
            a.partial_cmp(&b),
            Some(std::cmp::Ordering::Equal),
            "无值之间可比且相等"
        );

        let same = Time::new(None, rat(1, 2));
        assert_eq!(a, same);
        assert_eq!(a.partial_cmp(&same), Some(std::cmp::Ordering::Equal));

        let later = Time::new(Some(3), rat(1, 2));
        assert_eq!(a.partial_cmp(&later), Some(std::cmp::Ordering::Less));
        assert_eq!(later.partial_cmp(&a), Some(std::cmp::Ordering::Greater));

        // 同一时刻、时间基不同 —— 秒是唯一比较键
        let same_instant = rat(1, 4);
        let later_other_base = Time::new(Some(6), same_instant);
        assert_eq!(later, later_other_base);
        assert_eq!(
            later.partial_cmp(&later_other_base),
            Some(std::cmp::Ordering::Equal)
        );

        let earlier = Time::new(Some(1), rat(1, 4));
        assert_eq!(
            earlier.partial_cmp(&later_other_base),
            Some(std::cmp::Ordering::Less)
        );
        assert!(earlier < later_other_base);
    }

    /// "无值"的三种写法必须口径一致：零时间基（FFmpeg 自己的 `0/0`）与
    /// `None` / `AV_NOPTS_VALUE` 一样没有可用值 —— `has_value`、`into_value`、
    /// 秒换算、`Display` 与比较全部以 [`Time::instant`] 为准。
    #[test]
    fn test_zero_time_base_is_no_value() {
        let zero_base = Time::new(Some(5), Rational::ZERO);

        assert!(
            !zero_base.has_value(),
            "a zero time base leaves nothing to interpret"
        );
        assert_eq!(zero_base.into_value(), None);
        assert_eq!(zero_base.as_secs_f64(), 0.0);
        assert_eq!(zero_base.to_string(), "none");
        // 因此它与"完全没有值"是同一个时刻（排序上并列最前）
        assert_eq!(zero_base, Time::new(None, TIME_BASE));

        // 契约的另一半：正常时间基下 0 也是有值（0 秒），与"无值"不同。
        let zero = Time::new(Some(0), TIME_BASE);
        assert!(zero.has_value());
        assert_ne!(zero, Time::new(None, TIME_BASE));
    }

    /// 比较必须是**精确**的：同一时刻的两条不同计算路径在 `f64` 下可能相差 1 ulp，
    /// 而 `Eq` 要求传递性，浮点比较给不了。
    ///
    /// 这两组 `(time, time_base)` 在有理数上完全相等（后者 = 前者 × 43/43），
    /// 但 `time as f64 * time_base.as_f64()` 的两次舍入让它们相差一个 ulp
    /// （实测 `0x1.a653df01b3eb5p+27` vs `0x1.a653df01b3eb4p+27`）。
    #[test]
    fn test_equality_is_exact_not_floating_point() {
        let a = Time::new(Some(502_765), rat(1_804_821_558, 4_098_075));
        let b = Time::new(Some(21_618_895), rat(1_804_821_558, 176_217_225));

        // 数学上同一个时刻（交叉相乘相等）
        let lhs = i128::from(502_765) * i128::from(1_804_821_558) * i128::from(176_217_225);
        let rhs = i128::from(21_618_895) * i128::from(1_804_821_558) * i128::from(4_098_075);
        assert_eq!(lhs, rhs, "测试用的两组值必须真的表示同一时刻");
        // 而换算成 f64 后并不相等 —— 这正是旧实现会判它们不等的原因
        assert_ne!(
            a.as_secs_f64(),
            b.as_secs_f64(),
            "这个断言是前提：两条路径的 f64 结果相差 1 ulp"
        );

        assert_eq!(a, b, "同一时刻即使 f64 相差 1 ulp 也必须相等");
        assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Equal));
    }

    /// `Eq` 的传递性：三条彼此相等的链，任一两两比较都必须相等（浮点键做不到）。
    #[test]
    fn test_equality_is_transitive() {
        let a = Time::new(Some(1), rat(1, 3));
        let b = Time::new(Some(2), rat(1, 6));
        let c = Time::new(Some(3), rat(1, 9));
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "a == b 且 b == c ⇒ a == c");
    }

    /// `Display` 打印秒数且不会因 `time * time_base.num` 溢出 `i64` 而 panic。
    #[test]
    fn test_display_prints_seconds_without_overflow() {
        assert_eq!(Time::new(Some(2), rat(1, 2)).to_string(), "1 secs");
        assert_eq!(Time::new(None, TIME_BASE).to_string(), "none");
        // 旧实现会计算 `time_base.num as i64 * time`，在 debug 下 panic
        let huge = Time::new(Some(i64::MAX / 2), rat(1_000_000, 1_000_000));
        assert!(huge.to_string().ends_with(" secs"));
    }

    #[test]
    fn test_aligned_with() -> Result<()> {
        let a = Time::from_units(3, 16)?;
        let b = Time::from_units(1, 8)?;
        let aligned = a.aligned_with(b);
        assert_eq!(aligned.lhs, Some(3));
        assert_eq!(aligned.rhs, Some(2));
        Ok(())
    }

    #[test]
    fn test_into_aligned_with() -> Result<()> {
        let a = Time::from_units(2, 7)?;
        let b = Time::from_units(2, 3)?;
        let aligned = a.aligned_with(b);
        assert_eq!(aligned.lhs, Some(2));
        assert_eq!(aligned.rhs, Some(5));
        Ok(())
    }

    #[test]
    fn test_as_secs() -> Result<()> {
        let time = Time::from_nth_of_a_second(4)?;
        assert_eq!(time.as_secs(), 0.25);
        let time = Time::from_secs(0.3);
        assert_eq!(time.as_secs(), 0.3);
        let time = Time::new(None, rat(0, 0));
        assert_eq!(time.as_secs(), 0.0);
        Ok(())
    }

    #[test]
    fn test_as_secs_f64() -> Result<()> {
        let time = Time::from_nth_of_a_second(4)?;
        assert_eq!(time.as_secs_f64(), 0.25);
        let time = Time::from_secs_f64(0.3);
        assert_eq!(time.as_secs_f64(), 0.3);
        let time = Time::new(None, rat(0, 0));
        assert_eq!(time.as_secs_f64(), 0.0);
        Ok(())
    }

    #[test]
    fn test_into_value_none() {
        let time = Time::new(None, rat(0, 0));
        assert_eq!(time.into_value(), None);
    }

    #[test]
    fn test_add() {
        let a = Time::from_secs(0.2);
        let b = Time::from_secs(0.3);
        assert_eq!(a.aligned_with(b).add(), Time::from_secs(0.5));
    }

    #[test]
    fn test_subtract() {
        let a = Time::from_secs(0.8);
        let b = Time::from_secs(0.4);
        assert_eq!(a.aligned_with(b).subtract(), Time::from_secs(0.4));
    }

    #[test]
    fn test_apply() {
        let a = Time::from_secs(2.0);
        let b = Time::from_secs(0.25);
        assert_eq!(
            a.aligned_with(b).apply(|x, y| (2 * x) + (3 * y)),
            Time::from_secs(4.75)
        );
    }

    #[test]
    fn test_apply_different_time_bases() -> Result<()> {
        let a = Time::new(Some(3), rat(2, 32));
        let b = Time::from_nth_of_a_second(4)?;
        assert!(
            (a.aligned_with(b).apply(|x, y| x + y).as_secs()
                - Time::from_secs(7.0 / 16.0).as_secs())
            .abs()
                < 0.001
        );
        Ok(())
    }

    #[test]
    fn test_negative_into_duration_clamps() {
        assert_eq!(
            Duration::from(Time::new(Some(-100), rat(0, 0))),
            Duration::ZERO,
        )
    }

    /// `AV_NOPTS_VALUE` 是"无时间戳"，在 FFI 边界归一为 `None`：所有取值口都
    /// 按"无值"处理（早先 `into_value` 会把哨兵原样吐出来，与 `has_value` 矛盾）。
    #[test]
    fn test_av_no_pts_value_is_normalized() {
        let nopts = Time::new(Some(ffi::AV_NOPTS_VALUE), rat(0, 0));
        assert_eq!(nopts.time, None);
        assert!(!nopts.has_value());
        assert_eq!(nopts.into_value(), None);
        assert_eq!(nopts.as_secs_f64(), 0.0);
        assert_eq!(Duration::from(nopts).as_secs_f32(), 0.0);

        // 公开字段仍可手工塞入哨兵，取值口同样归一（不 panic、不吐出约 -9.2e13 秒）
        let hand_built = Time {
            time: Some(ffi::AV_NOPTS_VALUE),
            time_base: TIME_BASE,
        };
        assert!(!hand_built.has_value());
        assert_eq!(hand_built.into_value(), None);
        assert_eq!(hand_built.as_secs_f64(), 0.0);
    }
}
