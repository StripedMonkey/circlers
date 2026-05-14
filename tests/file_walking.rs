use std::{
    collections::BTreeSet,
    ffi::OsStr,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use circlers::walker::Walker;
use tempfile::TempDir;

pub enum FileKind {
    Dir,
    EmptyFile,
}

/// Build a temporary file tree from a declarative list of relative paths.
///
/// Paths ending with `/` or paired with `FileKind::Dir` become directories; all others become empty regular files (with
/// any missing parent directories created automatically).
///
/// # Example
/// ```
/// let tree = build_tree(&[
///     ("a/b/", FileKind::Dir),
///     ("a/b/file.txt", FileKind::EmptyFile),
/// ]).unwrap();
/// ```
fn build_tree(spec: &[(&str, FileKind)]) -> std::io::Result<TempDir> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    for (rel, kind) in spec {
        let path = root.join(rel);
        match kind {
            FileKind::Dir => std::fs::create_dir_all(&path)?,
            FileKind::EmptyFile => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, b"")?;
            }
        }
    }
    Ok(tmp)
}

// ── Walk helpers ─────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct WalkResult {
    /// Absolute paths of non-directory entries reported to `on_file`.
    files: BTreeSet<PathBuf>,
    /// Absolute paths of directory entries reported to `on_dir`.
    dirs: BTreeSet<PathBuf>,
}

/// Walk `root` synchronously and collect all reported entries.
///
/// Paths are reconstructed from the `dir_path` + entry name passed to each callback.
fn walk_sync(root: &Path) -> WalkResult {
    let result = Arc::new(Mutex::new(WalkResult::default()));

    let files_ref = result.clone();
    let dirs_ref = result.clone();

    let mut walker = Walker::new(
        move |_fd, dir_path, entry| {
            let mut path = PathBuf::from(OsStr::from_bytes(dir_path.to_bytes()));
            path.push(OsStr::from_bytes(entry.file_name().to_bytes()));
            files_ref.lock().unwrap().files.insert(path);
            Ok(())
        },
        move |_fd, dir_path, entry| {
            let mut path = PathBuf::from(OsStr::from_bytes(dir_path.to_bytes()));
            path.push(OsStr::from_bytes(entry.file_name().to_bytes()));
            dirs_ref.lock().unwrap().dirs.insert(path);
            Ok(())
        },
    );

    smol::block_on(walker.walk(root)).expect("walk failed");

    let guard = result.lock().unwrap();
    WalkResult {
        files: guard.files.clone(),
        dirs: guard.dirs.clone(),
    }
}

/// Assert that `result` contains exactly `expected_files` and `expected_dirs` (as paths relative to `root`).
fn assert_walk_eq(
    root: &Path,
    result: &WalkResult,
    expected_files: &[&str],
    expected_dirs: &[&str],
) {
    let expected_f: BTreeSet<PathBuf> = expected_files.iter().map(|p| root.join(p)).collect();
    let expected_d: BTreeSet<PathBuf> = expected_dirs.iter().map(|p| root.join(p)).collect();
    assert_eq!(result.files, expected_f, "files mismatch");
    assert_eq!(result.dirs, expected_d, "dirs mismatch");
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn empty_directory() {
    let tree = build_tree(&[]).unwrap();
    let result = walk_sync(tree.path());
    assert_walk_eq(tree.path(), &result, &[], &[]);
}

#[test]
fn flat_files_only() {
    let tree = build_tree(&[
        ("a.txt", FileKind::EmptyFile),
        ("b.txt", FileKind::EmptyFile),
        ("c.txt", FileKind::EmptyFile),
    ])
    .unwrap();
    let result = walk_sync(tree.path());
    assert_walk_eq(tree.path(), &result, &["a.txt", "b.txt", "c.txt"], &[]);
}

#[test]
fn deep_single_chain() {
    let tree = build_tree(&[
        ("a/b/c/", FileKind::Dir),
        ("a/b/c/leaf.txt", FileKind::EmptyFile),
    ])
    .unwrap();
    let result = walk_sync(tree.path());
    assert_walk_eq(
        tree.path(),
        &result,
        &["a/b/c/leaf.txt"],
        &["a", "a/b", "a/b/c"],
    );
}

#[test]
fn wide_many_subdirs() {
    let spec: Vec<(&str, FileKind)> = (0..8)
        .flat_map(|i| {
            let dir = Box::leak(format!("sub{i}").into_boxed_str()) as &str;
            let file = Box::leak(format!("sub{i}/file.txt").into_boxed_str()) as &str;
            [(dir, FileKind::Dir), (file, FileKind::EmptyFile)]
        })
        .collect();
    let tree = build_tree(&spec).unwrap();
    let result = walk_sync(tree.path());

    let expected_dirs: Vec<&str> = (0..8)
        .map(|i| Box::leak(format!("sub{i}").into_boxed_str()) as &str)
        .collect();
    let expected_files: Vec<&str> = (0..8)
        .map(|i| Box::leak(format!("sub{i}/file.txt").into_boxed_str()) as &str)
        .collect();

    assert_walk_eq(tree.path(), &result, &expected_files, &expected_dirs);
}

#[test]
fn mixed_tree() {
    let tree = build_tree(&[
        ("a/", FileKind::Dir),
        ("a/a1.txt", FileKind::EmptyFile),
        ("a/b/", FileKind::Dir),
        ("a/b/b1.txt", FileKind::EmptyFile),
        ("a/b/b2.txt", FileKind::EmptyFile),
        ("a/c/", FileKind::Dir),
        ("d.txt", FileKind::EmptyFile),
    ])
    .unwrap();
    let result = walk_sync(tree.path());
    assert_walk_eq(
        tree.path(),
        &result,
        &["a/a1.txt", "a/b/b1.txt", "a/b/b2.txt", "d.txt"],
        &["a", "a/b", "a/c"],
    );
}

#[test]
fn split_work_yields_valid_paths() {
    let tree = build_tree(&[
        ("x/", FileKind::Dir),
        ("y/", FileKind::Dir),
        ("z/", FileKind::Dir),
    ])
    .unwrap();

    let root = tree.path();
    let files_seen = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
    let dirs_seen = Arc::new(Mutex::new(Vec::<PathBuf>::new()));

    let files_ref = files_seen.clone();
    let dirs_ref = dirs_seen.clone();

    let mut walker = Walker::new(
        move |_fd, dir_path, entry| {
            let mut p = PathBuf::from(OsStr::from_bytes(dir_path.to_bytes()));
            p.push(OsStr::from_bytes(entry.file_name().to_bytes()));
            files_ref.lock().unwrap().push(p);
            Ok(())
        },
        move |_fd, dir_path, entry| {
            let mut p = PathBuf::from(OsStr::from_bytes(dir_path.to_bytes()));
            p.push(OsStr::from_bytes(entry.file_name().to_bytes()));
            dirs_ref.lock().unwrap().push(p);
            Ok(())
        },
    );

    smol::block_on(async {
        // Seed root without draining.
        walker.work_queue().extend_cold([root]).await;

        // Extract some work before draining; paths must be stat-able directories.
        let split = walker.work_queue().try_split_work(2).await;
        for path in &split {
            let path = Path::new(OsStr::from_bytes(path.to_bytes()));
            assert!(path.is_dir(), "split_work returned a non-directory: {path:?}");
        }

        // Re-inject split paths and drain everything.
        walker.work_queue().extend_cold(
            split.iter().map(|p| Path::new(OsStr::from_bytes(p.to_bytes())))
        ).await;
        walker.drain().await.expect("drain failed");
    });
}
