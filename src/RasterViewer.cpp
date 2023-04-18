#include "SDLViewer.h"

#include <Eigen/Core>

#include <functional>
#include <iostream>

#include "raster.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Frame Size
const int width = 500;
const int height = 500;

// Orthographic and Perspective Projection
const double near = 0.1;
const double far = 100;

std::vector<VertexAttributes> vertices;

bool insert_mode = false;
bool translate_mode = false;
bool delete_mode = false;

Eigen::Matrix4d get_camera_transformation(Eigen::Vector3d camera_position)
{
    // Calculate the look-at and view direction
    Eigen::Vector3d gaze_direction = camera_position;
    Eigen::Vector3d view_up = Eigen::Vector3d(0.0, 1.0, 0.0);

    // Calculate the camera reference system
    Eigen::Vector3d w = -gaze_direction.normalized();
    Eigen::Vector3d u = view_up.cross(w).normalized();
    Eigen::Vector3d v = w.cross(u);

    Eigen::Matrix4d camera_transform;
    camera_transform << u(0), v(0), w(0), camera_position(0),
        u(1), v(1), w(1), camera_position(1),
        u(2), v(2), w(2), camera_position(2),
        0, 0, 0, 1;

    return camera_transform.inverse();
}

Eigen::Matrix4d get_perspective_projection()
{
    Eigen::Matrix4d perspective_projection;

    perspective_projection << near, 0, 0, 0,
        0, near, 0, 0,
        0, 0, near + far, -(far * near),
        0, 0, 1, 0;

    return perspective_projection;
}

Eigen::Matrix4d get_orthographic_projection(double top, double right)
{
    // Specifying the bounding box coordinates for canonical cube
    double left = -right;
    double bottom = -top;

    Eigen::Matrix4d ortho_projection;
    ortho_projection << 2 / (right - left), 0, 0, -(right + left) / (right - left),
        0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom),
        0, 0, 2 / (near - far), -(near + far) / (near - far),
        0, 0, 0, 1;

    return ortho_projection;
}

Eigen::Matrix4d get_view_transformation(double aspect_ratio)
{
    Eigen::Matrix4d view = Eigen::Matrix4d::Identity();

    if (aspect_ratio < 1)
        view(0, 0) = aspect_ratio;
    else
        view(1, 1) = 1 / aspect_ratio;

    return view;
}

Eigen::Vector4d get_canonical_coordinates(int x_screen, int y_screen)
{
    return Eigen::Vector4d((double(x_screen) / double(width) * 2) - 1, (double(height - 1 - y_screen) / double(height) * 2) - 1, 0, 1);
}

int main(int argc, char *args[])
{
    // The Framebuffer storing the image rendered by the rasterizer
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(width, height);

    // Global Constants (empty in this example)
    UniformAttributes uniform;

    // Basic rasterization program
    Program program;

    // The vertex shader is the identity
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        VertexAttributes out;
        out.position = uniform.transformation * va.position;
        out.color = va.color;
        return out;
    };

    // The fragment shader uses a fixed color
    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        FragmentAttributes out(va.color(0), va.color(1), va.color(2), va.color(3));
        out.position = va.position;
        return out;
    };

    // The blending shader converts colors between 0 and 1 to uint8
    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous)
    {
        if (fa.position[2] <= previous.depth)
        {
            FrameBufferAttributes out(fa.color[0] * 255, fa.color[1] * 255, fa.color[2] * 255, fa.color[3] * 255);
            out.depth = fa.position[2];
            return out;
        }
        else
            return previous;
    };

    // Add a transformation to compensate for the aspect ratio of the framebuffer
    double aspect_ratio = double(frameBuffer.cols()) / double(frameBuffer.rows());
    double top = 16;
    double right = top;

    // Orthographic Projection
    // Set the transformations for camera space and orthographic projection
    uniform.camera_position = Eigen::Vector3d(0, 0, -5);

    Eigen::Matrix4d camera_transformation = get_camera_transformation(uniform.camera_position);
    Eigen::Matrix4d perspective_projection = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d ortho_projection = get_orthographic_projection(top, right);
    Eigen::Matrix4d view = get_view_transformation(aspect_ratio);

    uniform.transformation = view * ortho_projection * perspective_projection * camera_transformation;
    uniform.inverse_transformation = camera_transformation.inverse() * perspective_projection.inverse() * ortho_projection.inverse() * view.inverse();

    // World Coordinates
    vertices.push_back(VertexAttributes(-16, -16, 45.05));
    vertices.push_back(VertexAttributes(16, -16, 45.05));
    vertices.push_back(VertexAttributes(0, 16, 45.05));
    vertices[0].color << 1, 0, 0, 1;
    vertices[1].color << 0, 1, 0, 1;
    vertices[2].color << 0, 0, 1, 1;

    // Initialize the viewer and the corresponding callbacks
    SDLViewer viewer;
    viewer.init("Viewer Example", width, height);

    viewer.mouse_move = [](int x, int y, int xrel, int yrel) {
    };

    viewer.mouse_pressed = [&](int x, int y, bool is_pressed, int button, int clicks)
    {
        Eigen::Vector4d canonical_coords = get_canonical_coordinates(x, y);
        Eigen::Vector4d world_coords = uniform.inverse_transformation * canonical_coords;
        if (insert_mode)
        {
            vertices[2].position = world_coords;
            viewer.redraw_next = true;
        }
    };

    viewer.mouse_wheel = [&](int dx, int dy, bool is_direction_normal) {
    };

    viewer.key_pressed = [&](char key, bool is_pressed, int modifier, int repeat)
    {
        switch (key)
        {
        case 'i':
            insert_mode = true;
            break;
        case 'o':
            translate_mode = true;
            break;
        case 'p':
            delete_mode = true;
            break;
        case 'z':
            insert_mode = false;
            translate_mode = false;
            delete_mode = false;
            break;
        }
    };

    viewer.redraw = [&](SDLViewer &viewer)
    {
        // Clear the framebuffer
        for (unsigned i = 0; i < frameBuffer.rows(); i++)
            for (unsigned j = 0; j < frameBuffer.cols(); j++)
                frameBuffer(i, j).color << 0, 0, 0, 1;

        rasterize_triangles(program, uniform, vertices, frameBuffer);

        // Buffer for exchanging data between rasterizer and sdl viewer
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

        for (unsigned i = 0; i < frameBuffer.rows(); i++)
        {
            for (unsigned j = 0; j < frameBuffer.cols(); j++)
            {
                R(i, frameBuffer.cols() - 1 - j) = frameBuffer(i, j).color(0);
                G(i, frameBuffer.cols() - 1 - j) = frameBuffer(i, j).color(1);
                B(i, frameBuffer.cols() - 1 - j) = frameBuffer(i, j).color(2);
                A(i, frameBuffer.cols() - 1 - j) = frameBuffer(i, j).color(3);
            }
        }
        viewer.draw_image(R, G, B, A);
    };

    viewer.launch();

    return 0;
}
