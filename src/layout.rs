//! The shape a decoded frame's samples occupy — shared by audio and video.
//!
//! This is the crate's leaf type for frame geometry: it depends on nothing else
//! here, which is what lets both [`PixelFormat`](crate::pixel::PixelFormat) and
//! [`SampleFormat`](crate::fmt::SampleFormat) describe their data with it without
//! the two modules depending on each other.
//!
//! Re-exported as [`crate::DataLayout`] and, for historical paths, as
//! `crate::fmt::DataLayout`.

/// The data layout a media format requires.
///
/// **Media-agnostic**: audio and video both use it, and it is reached through
/// [`PixelFormat::data_layout`](crate::pixel::PixelFormat::data_layout) as well as
/// [`SampleFormat::data_layout`](crate::fmt::SampleFormat::data_layout). The split
/// is by memory layout, not media kind — a packed video format (e.g. `RGB24`) and
/// a packed audio format (e.g. `S16`) are both [`Interleaved`](Self::Interleaved),
/// while a planar video format (e.g. `YUV420P`) and a planar audio format (e.g.
/// `FLTP`) are both [`Planar`](Self::Planar). Nothing here distinguishes the two
/// media kinds.
///
/// Every plane is a row-major `rows x cols` block; an interleaved layout
/// additionally carries a `components` axis, which [`Self::shapes`] keeps
/// separate and [`Self::plane_row_extent`] folds into the columns. Audio
/// layouts are always one row tall, so the row stride never takes effect
/// for them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataLayout {
    /// One interleaved array of shape `(rows, cols, components)`.
    Interleaved {
        /// Rows of the array — the picture height for video, `1` for audio.
        rows: usize,
        /// Columns of the array — the sample count for audio; for video, the
        /// picture width rounded up to whole row units, as FFmpeg's linesize
        /// does (a horizontally subsampled packed format such as `YUYV422`
        /// rounds an odd width up to a whole 2-pixel unit).
        cols: usize,
        /// Storage elements per row-unit: the per-pixel element run for packed
        /// video, the channel count for packed audio.
        components: usize,
    },
    /// One array per plane, each of shape `(rows, cols)`.
    ///
    /// A video plane holds a single component (or several, for a semi-planar
    /// chroma plane such as `NV12`'s), subsampled planes carrying their own
    /// smaller size; an audio plane holds one channel.
    Planar(Vec<(usize, usize)>),
}

impl DataLayout {
    /// Number of arrays this layout is stored as.
    pub fn num_planes(&self) -> usize {
        match self {
            Self::Interleaved { .. } => 1,
            Self::Planar(planes) => planes.len(),
        }
    }

    /// Plane `plane` as a flat `(rows, samples_per_row)` extent.
    ///
    /// The component axis of an interleaved layout is folded into
    /// `samples_per_row`, which is exactly what a row-wise copy of the plane
    /// needs. Note the difference from [`Self::shapes`], which reports the array
    /// shape [`FrameData`](crate::frame::FrameData) stores (components kept as a
    /// separate axis).
    pub fn plane_row_extent(&self, plane: usize) -> Option<(usize, usize)> {
        match self {
            Self::Interleaved {
                rows,
                cols,
                components,
            } => (plane == 0).then(|| (*rows, cols * components)),
            Self::Planar(planes) => planes.get(plane).copied(),
        }
    }

    /// The shape of the array [`FrameData`](crate::frame::FrameData) stores plane
    /// `plane` as: the counterpart of [`Self::plane_row_extent`] that keeps the
    /// component axis separate instead of folding it into the columns.
    fn plane_shape(&self, plane: usize) -> Option<(usize, usize)> {
        match self {
            Self::Interleaved { rows, cols, .. } => (plane == 0).then_some((*rows, *cols)),
            Self::Planar(planes) => planes.get(plane).copied(),
        }
    }

    /// Every plane's shape, for diagnostics.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        (0..self.num_planes())
            .filter_map(|plane| self.plane_shape(plane))
            .collect()
    }
}
