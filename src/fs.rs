//! What `open()` does, as decided by the host.
//!
//! The interpreter never touches the filesystem itself: `open(path)` calls
//! the [`Opener`] in the [`Config`](crate::Config), and the script reads
//! whatever [`Read`] it hands back. Scripts can't tell a real file from an
//! in-memory buffer or anything else the host provides.

use std::fmt;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use cap_std::ambient_authority;
use cap_std::fs::Dir;

/// A readable stream handed to a script by `open()`.
pub type Reader = Box<dyn Read + Send>;

type OpenFn = dyn Fn(&Path) -> io::Result<Reader> + Send + Sync;

/// The host's implementation of `open()`.
#[derive(Clone, Default)]
pub struct Opener(Option<Arc<OpenFn>>);

impl Opener {
    /// `open()` always fails with a `PermissionError`. The default.
    pub fn deny() -> Opener {
        Opener(None)
    }

    /// `open()` reads any file the process can.
    pub fn filesystem() -> Opener {
        Opener::custom(|p| Ok(Box::new(std::fs::File::open(p)?) as Reader))
    }

    /// `open()` may only read files allowed by `list`.
    pub fn allowlist(list: Allowlist) -> Opener {
        Opener::custom(move |p| list.open(p))
    }

    /// `open()` is entirely up to `f`. An `io::ErrorKind::PermissionDenied`
    /// error surfaces as a script `PermissionError`, anything else as an
    /// `IOError`.
    pub fn custom(f: impl Fn(&Path) -> io::Result<Reader> + Send + Sync + 'static) -> Opener {
        Opener(Some(Arc::new(f)))
    }

    /// `None` if `open()` is disabled.
    pub(crate) fn open(&self, path: &Path) -> Option<io::Result<Reader>> {
        self.0.as_ref().map(|f| f(path))
    }
}

impl fmt::Debug for Opener {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.is_some() {
            "Opener(..)"
        } else {
            "Opener::deny()"
        })
    }
}

/// Files and directory trees that `open()` may read.
///
/// Directories are opened when they are added, and every later lookup is
/// resolved *beneath* that handle (via `cap-std`): `..` components and
/// symlinks cannot escape it, even if the filesystem changes afterwards. A
/// file entry allows exactly that name inside its directory, which is pinned
/// the same way.
///
/// Requested paths are matched against entries by their absolute path
/// (relative paths are taken relative to the current directory), without
/// resolving symlinks.
#[derive(Default)]
pub struct Allowlist {
    dirs: Vec<(PathBuf, Dir)>,
    files: Vec<(PathBuf, Dir, PathBuf)>,
}

impl Allowlist {
    pub fn new() -> Allowlist {
        Allowlist::default()
    }

    /// Allow every file beneath `path`.
    pub fn dir(mut self, path: impl AsRef<Path>) -> io::Result<Allowlist> {
        let path = absolute(path.as_ref())?;
        let dir = Dir::open_ambient_dir(&path, ambient_authority())?;
        self.dirs.push((path, dir));
        Ok(self)
    }

    /// Allow exactly the file at `path`.
    pub fn file(mut self, path: impl AsRef<Path>) -> io::Result<Allowlist> {
        let path = absolute(path.as_ref())?;
        let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} is not a file path", path.display()),
            ));
        };
        let dir = Dir::open_ambient_dir(parent, ambient_authority())?;
        let name = PathBuf::from(name);
        self.files.push((path, dir, name));
        Ok(self)
    }

    fn open(&self, requested: &Path) -> io::Result<Reader> {
        let path = absolute(requested)?;
        for (allowed, dir, name) in &self.files {
            if *allowed == path {
                return Ok(Box::new(dir.open(name)?));
            }
        }
        for (root, dir) in &self.dirs {
            if let Ok(rel) = path.strip_prefix(root)
                && rel.components().next().is_some()
            {
                return Ok(Box::new(dir.open(rel)?));
            }
        }
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{} is not in the allowlist", requested.display()),
        ))
    }
}

/// Make `p` absolute against the current directory and drop `.` components.
/// `..` is kept, so it is resolved (and confined) by `cap-std`.
fn absolute(p: &Path) -> io::Result<PathBuf> {
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()?.join(p)
    };
    Ok(joined
        .components()
        .filter(|c| *c != Component::CurDir)
        .collect())
}
