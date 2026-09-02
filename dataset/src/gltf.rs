#[cfg(test)]
mod test {
    use std::println;

    #[test]
    fn test_gltf() {
        let gltf = gltf::Gltf::open("/Users/joe/Desktop/fern.glb").unwrap();

        for (i, extension) in gltf.extensions_used().enumerate() {
            println!("extension_used[{}]: {}", i, extension);
        }

        for (i, extension) in gltf.extensions_required().enumerate() {
            println!("extension_required[{}]: {}", i, extension);
        }

        for (i, camera) in gltf.cameras().enumerate() {
            println!("camera[{}]: {}", i, camera.name().unwrap_or_default());
        }

        for (i, scene) in gltf.scenes().enumerate() {
            println!("scene[{}]: {}", i, scene.name().unwrap_or_default());
            for (i, node) in scene.nodes().enumerate() {
                println!(" node[{}]: {}", i, node.name().unwrap_or_default());
                for (i, child) in node.children().enumerate() {
                    println!("  child[{}]: {}", i, child.name().unwrap_or_default());
                    if child.camera().is_some() {
                        println!("   found camera");
                    }
                    if let Some(mesh) = child.mesh() {
                        println!("   found mesh");
                    }
                }
            }
        }
    }
}
