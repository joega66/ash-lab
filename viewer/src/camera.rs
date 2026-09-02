use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Rotation3, SquareMatrix};

pub const OPENGL_TO_VULKAN_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
    cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, -1.0, 0.0, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
    cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
);

pub struct Camera {
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn default(width: u32, height: u32) -> Self {
        let eye = cgmath::Point3::new(0.0, 1.0, 2.0);
        let center = cgmath::Point3::new(0.0, 0.0, -1.0);
        let view = cgmath::Matrix4::look_at_rh(eye, center, cgmath::Vector3::unit_y())
            .invert()
            .unwrap();
        let rotation = cgmath::Quaternion::from(cgmath::Matrix3::from_cols(
            view.x.truncate(),
            view.y.truncate(),
            view.z.truncate(),
        ));
        let position = view.w.truncate();
        Self {
            position: position,
            rotation: rotation,
            aspect: width as f32 / height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    pub fn forward(&self) -> cgmath::Vector3<f32> {
        self.rotation * cgmath::Vector3::<f32>::new(0.0, 0.0, -1.0)
    }

    pub fn _up(&self) -> cgmath::Vector3<f32> {
        self.rotation * cgmath::Vector3::<f32>::new(0.0, 1.0, 0.0)
    }

    pub fn right(&self) -> cgmath::Vector3<f32> {
        self.rotation * cgmath::Vector3::<f32>::new(1.0, 0.0, 0.0)
    }

    pub fn rotate(&mut self, x_offset: f32, y_offset: f32) {
        let yaw =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), cgmath::Rad(-x_offset));
        let pitch =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_x(), cgmath::Rad(-y_offset));
        self.rotation = (yaw * self.rotation * pitch).normalize();
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = (cgmath::Matrix4::from_translation(self.position)
            * cgmath::Matrix4::from(self.rotation))
        .invert()
        .unwrap();

        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_VULKAN_MATRIX * proj * view;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}
impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
