/**
 * ComfyUI-DiT360 Interactive 360° Viewer
 *
 * Adds a custom widget to the Equirect360Viewer node that displays
 * panoramic images using Three.js for interactive 360° navigation.
 */

import { app } from "../../scripts/app.js";

// Three.js CDN URL
const THREE_JS_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

let THREE = null;
let isThreeJSLoaded = false;

/**
 * Load Three.js dynamically from CDN
 */
async function loadThreeJS() {
    if (isThreeJSLoaded && THREE) {
        return THREE;
    }

    try {
        THREE = await import(THREE_JS_URL);
        isThreeJSLoaded = true;
        console.log("✅ Three.js loaded successfully for 360° viewer");
        return THREE;
    } catch (error) {
        console.error("❌ Failed to load Three.js:", error);
        alert("Failed to load Three.js for 360° viewer. Check your internet connection.");
        return null;
    }
}

/**
 * Open 360° viewer modal
 */
async function open360Viewer(imageDataURL) {
    // Load Three.js if not already loaded
    if (!THREE) {
        await loadThreeJS();
        if (!THREE) return;  // Failed to load
    }

    // Create modal overlay
    const modal = document.createElement("div");
    modal.id = "equirect360-viewer-modal";
    modal.style.cssText = `
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(0, 0, 0, 0.95);
        z-index: 10000;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    `;

    // Create canvas for Three.js
    const canvas = document.createElement("canvas");
    canvas.style.cssText = `
        width: 90vw;
        height: 90vh;
        cursor: grab;
    `;
    modal.appendChild(canvas);

    // Create controls overlay
    const controls = document.createElement("div");
    controls.style.cssText = `
        position: absolute;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 20px;
        border-radius: 5px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 14px;
        pointer-events: none;
    `;
    controls.innerHTML = `
        <strong>🌐 360° Panorama Viewer</strong><br>
        <small>Drag to rotate • Scroll to zoom • ESC or click button to close</small>
    `;
    modal.appendChild(controls);

    // Create close button
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕ Close";
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px; right: 20px;
        padding: 10px 20px;
        background: #fff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        font-size: 14px;
    `;
    closeBtn.onclick = () => cleanup();
    modal.appendChild(closeBtn);

    document.body.appendChild(modal);

    // Initialize Three.js scene
    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
        75,  // FOV
        canvas.clientWidth / canvas.clientHeight,  // Aspect
        0.1,  // Near
        1000  // Far
    );
    camera.position.set(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Create sphere for panorama (inverted for inside viewing)
    const geometry = new THREE.SphereGeometry(500, 60, 40);
    geometry.scale(-1, 1, 1);  // Invert X for inside viewing

    // Load panorama texture
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(imageDataURL, () => {
        console.log("✅ Panorama texture loaded");
    }, undefined, (error) => {
        console.error("❌ Failed to load panorama texture:", error);
    });

    const material = new THREE.MeshBasicMaterial({ map: texture });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    // Mouse/touch controls
    let isDragging = false;
    let previousMouse = { x: 0, y: 0 };
    let rotation = { x: 0, y: 0 };

    // Mouse down
    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        canvas.style.cursor = "grabbing";
        previousMouse = { x: e.clientX, y: e.clientY };
    });

    // Mouse move
    canvas.addEventListener("mousemove", (e) => {
        if (!isDragging) return;

        const deltaX = e.clientX - previousMouse.x;
        const deltaY = e.clientY - previousMouse.y;

        rotation.y -= deltaX * 0.005;
        rotation.x -= deltaY * 0.005;

        // Clamp vertical rotation to avoid flipping
        rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));

        previousMouse = { x: e.clientX, y: e.clientY };
    });

    // Mouse up
    canvas.addEventListener("mouseup", () => {
        isDragging = false;
        canvas.style.cursor = "grab";
    });

    canvas.addEventListener("mouseleave", () => {
        isDragging = false;
        canvas.style.cursor = "grab";
    });

    // Scroll for zoom (FOV adjustment)
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        camera.fov += e.deltaY * 0.05;
        camera.fov = Math.max(30, Math.min(120, camera.fov));
        camera.updateProjectionMatrix();
    });

    // Touch controls for mobile
    let touchStart = null;

    canvas.addEventListener("touchstart", (e) => {
        if (e.touches.length === 1) {
            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
    });

    canvas.addEventListener("touchmove", (e) => {
        if (e.touches.length === 1 && touchStart) {
            e.preventDefault();
            const deltaX = e.touches[0].clientX - touchStart.x;
            const deltaY = e.touches[0].clientY - touchStart.y;

            rotation.y -= deltaX * 0.005;
            rotation.x -= deltaY * 0.005;
            rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rotation.x));

            touchStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
    });

    canvas.addEventListener("touchend", () => {
        touchStart = null;
    });

    // Keyboard controls (ESC to close)
    const keyHandler = (e) => {
        if (e.key === "Escape") {
            cleanup();
        }
    };
    document.addEventListener("keydown", keyHandler);

    // Click outside to close
    modal.addEventListener("click", (e) => {
        if (e.target === modal) {
            cleanup();
        }
    });

    // Animation loop
    let animationId;
    function animate() {
        animationId = requestAnimationFrame(animate);

        // Apply rotation to camera
        camera.rotation.order = "YXZ";
        camera.rotation.y = rotation.y;
        camera.rotation.x = rotation.x;

        renderer.render(scene, camera);
    }

    animate();

    // Cleanup function
    function cleanup() {
        cancelAnimationFrame(animationId);
        renderer.dispose();
        geometry.dispose();
        material.dispose();
        texture.dispose();
        document.removeEventListener("keydown", keyHandler);
        if (document.body.contains(modal)) {
            document.body.removeChild(modal);
        }
        console.log("✅ 360° viewer closed");
    }

    // Handle window resize
    const resizeHandler = () => {
        if (document.body.contains(modal)) {
            camera.aspect = canvas.clientWidth / canvas.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        }
    };
    window.addEventListener("resize", resizeHandler);

    console.log("✅ 360° viewer opened");
}

/**
 * Register ComfyUI extension
 */
app.registerExtension({
    name: "ComfyUI.DiT360.Equirect360Viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Pre-load Three.js when the viewer node is registered
        if (nodeData.name === "Equirect360Viewer") {
            await loadThreeJS();
        }
    },

    async nodeCreated(node) {
        if (node.comfyClass === "Equirect360Viewer") {
            // Add custom widget for 360° viewing button
            const widget = node.addWidget("button", "🌐 View 360°", "view360", () => {
                // Get the latest image from the node
                if (node.imgs && node.imgs.length > 0) {
                    const imageData = node.imgs[0].src;
                    open360Viewer(imageData);
                } else {
                    alert("No panorama to view. Generate an image first!");
                }
            });

            widget.serialize = false;  // Don't save button state in workflow
        }
    }
});

console.log("✅ Equirect360Viewer extension loaded");
