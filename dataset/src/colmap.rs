use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
};

pub type CameraId = u32;
pub type ImageId = u32;
pub type Point3DId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraModel {
    SimplePinhole,
    Pinhole,
    SimpleRadial,
    Radial,
    OpenCV,
    OpenCVFisheye,
    FullOpenCV,
    Fov,
    SimpleRadialFisheye,
    RadialFisheye,
    ThinPrismFisheye,
}

impl CameraModel {
    fn from_id(id: i32) -> io::Result<Self> {
        Ok(match id {
            0 => CameraModel::SimplePinhole,
            1 => CameraModel::Pinhole,
            2 => CameraModel::SimpleRadial,
            3 => CameraModel::Radial,
            4 => CameraModel::OpenCV,
            5 => CameraModel::OpenCVFisheye,
            6 => CameraModel::FullOpenCV,
            7 => CameraModel::Fov,
            8 => CameraModel::SimpleRadialFisheye,
            9 => CameraModel::RadialFisheye,
            10 => CameraModel::ThinPrismFisheye,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown COLMAP camera model id {id}"),
                ));
            }
        })
    }

    fn num_params(self) -> usize {
        match self {
            CameraModel::SimplePinhole => 3,
            CameraModel::Pinhole => 4,
            CameraModel::SimpleRadial => 4,
            CameraModel::Radial => 5,
            CameraModel::OpenCV => 8,
            CameraModel::OpenCVFisheye => 8,
            CameraModel::FullOpenCV => 12,
            CameraModel::Fov => 5,
            CameraModel::SimpleRadialFisheye => 4,
            CameraModel::RadialFisheye => 5,
            CameraModel::ThinPrismFisheye => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub id: CameraId,
    pub model: CameraModel,
    pub width: u64,
    pub height: u64,
    pub params: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Image {
    pub id: ImageId,
    /// Rotation as a (w, x, y, z) quaternion, world-from-camera.
    pub qvec: [f64; 4],
    /// Translation, world-from-camera.
    pub tvec: [f64; 3],
    pub camera_id: CameraId,
    pub name: String,
    /// 2D keypoint locations in this image.
    pub xys: Vec<[f64; 2]>,
    /// Point3D id observed by each keypoint in `xys`, or -1 if unmatched.
    pub point3d_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct Point3D {
    pub id: Point3DId,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
    /// (image_id, index into that image's `xys`/`point3d_ids`) pairs observing this point.
    pub track: Vec<(ImageId, i32)>,
}

#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    pub cameras: HashMap<CameraId, Camera>,
    pub images: HashMap<ImageId, Image>,
    pub points3d: HashMap<Point3DId, Point3D>,
}

impl Reconstruction {
    /// Loads cameras.bin, images.bin, and points3D.bin from a COLMAP sparse
    /// reconstruction directory (e.g. `sparse/0`).
    pub fn load_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref();
        Ok(Self {
            cameras: read_cameras_bin(dir.join("cameras.bin"))?,
            images: read_images_bin(dir.join("images.bin"))?,
            points3d: read_points3d_bin(dir.join("points3D.bin"))?,
        })
    }
}

pub fn read_cameras_bin(path: impl AsRef<Path>) -> io::Result<HashMap<CameraId, Camera>> {
    let mut reader = open(path)?;
    let num_cameras = read_u64(&mut reader)?;
    let mut cameras = HashMap::with_capacity(num_cameras as usize);
    for _ in 0..num_cameras {
        let id = read_i32(&mut reader)? as CameraId;
        let model = CameraModel::from_id(read_i32(&mut reader)?)?;
        let width = read_u64(&mut reader)?;
        let height = read_u64(&mut reader)?;
        let mut params = vec![0.0; model.num_params()];
        for param in &mut params {
            *param = read_f64(&mut reader)?;
        }
        cameras.insert(
            id,
            Camera {
                id,
                model,
                width,
                height,
                params,
            },
        );
    }
    Ok(cameras)
}

pub fn read_images_bin(path: impl AsRef<Path>) -> io::Result<HashMap<ImageId, Image>> {
    let mut reader = open(path)?;
    let num_images = read_u64(&mut reader)?;
    let mut images = HashMap::with_capacity(num_images as usize);
    for _ in 0..num_images {
        let id = read_i32(&mut reader)? as ImageId;
        let qvec = [
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
        ];
        let tvec = [
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
        ];
        let camera_id = read_i32(&mut reader)? as CameraId;
        let name = read_null_terminated_string(&mut reader)?;

        let num_points2d = read_u64(&mut reader)?;
        let mut xys = Vec::with_capacity(num_points2d as usize);
        let mut point3d_ids = Vec::with_capacity(num_points2d as usize);
        for _ in 0..num_points2d {
            let x = read_f64(&mut reader)?;
            let y = read_f64(&mut reader)?;
            let point3d_id = read_i64(&mut reader)?;
            xys.push([x, y]);
            point3d_ids.push(point3d_id);
        }

        images.insert(
            id,
            Image {
                id,
                qvec,
                tvec,
                camera_id,
                name,
                xys,
                point3d_ids,
            },
        );
    }
    Ok(images)
}

pub fn read_points3d_bin(path: impl AsRef<Path>) -> io::Result<HashMap<Point3DId, Point3D>> {
    let mut reader = open(path)?;
    let num_points = read_u64(&mut reader)?;
    let mut points = HashMap::with_capacity(num_points as usize);
    for _ in 0..num_points {
        let id = read_u64(&mut reader)?;
        let xyz = [
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
            read_f64(&mut reader)?,
        ];
        let rgb = [
            read_u8(&mut reader)?,
            read_u8(&mut reader)?,
            read_u8(&mut reader)?,
        ];
        let error = read_f64(&mut reader)?;

        let track_length = read_u64(&mut reader)?;
        let mut track = Vec::with_capacity(track_length as usize);
        for _ in 0..track_length {
            let image_id = read_i32(&mut reader)? as ImageId;
            let point2d_idx = read_i32(&mut reader)?;
            track.push((image_id, point2d_idx));
        }

        points.insert(
            id,
            Point3D {
                id,
                xyz,
                rgb,
                error,
                track,
            },
        );
    }
    Ok(points)
}

fn open(path: impl AsRef<Path>) -> io::Result<BufReader<File>> {
    let path: PathBuf = path.as_ref().to_path_buf();
    File::open(&path)
        .map(BufReader::new)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_null_terminated_string(r: &mut impl Read) -> io::Result<String> {
    let mut bytes = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        r.read_exact(&mut byte)?;
        if byte[0] == 0 {
            break;
        }
        bytes.push(byte[0]);
    }
    String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}