#include "SDLViewer.h"

#include <Eigen/Core>

#include <functional>
#include <iostream>

#include "raster.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Frame Size
const int width = 1000;
const int height = 700;

// The Framebuffer storing the image rendered by the rasterizer
Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(width, height);

// Orthographic and Perspective Projection
const double near = 0.1;
const double far = 100;

// For highlighting triangle
int selected_index = -1;
bool is_mouse_pressed = false;

// Rotation angle in degrees
double angle = 10.0;

// For insertion preview, we need to keep separate line and triangle vertices
std::vector<VertexAttributes> triangle_vertices;
std::vector<VertexAttributes> line_vertices;

// Inverse transformation to transform from screen space to world space
Eigen::Matrix4d inverse_transformation;

// Model transformations (translation, rotation, and scaling) for each triangle
std::vector<Eigen::Matrix4d> model_transformations;

// Modes
enum mode
{
    NONE = 0,
    INSERT_MODE = 1,
    TRANSLATE_MODE = 2,
    DELETE_MODE = 3,
    COLOR_MODE = 4
};

mode current_mode;

// Colors
enum color
{
    BLACK = 0,
    RED = 1,
    BLUE = 2,
    GREEN = 3,
    PURPLE = 4,
    YELLOW = 5,
    GREY = 6,
    PINK = 7,
    ORANGE = 8,
    AQUA = 9
};

Eigen::Vector4d get_color_vector(color color_code)
{
    switch (color_code)
    {
    case BLACK:
        return Eigen::Vector4d(0, 0, 0, 1);
    case RED:
        return Eigen::Vector4d(1, 0, 0, 1);
    case BLUE:
        return Eigen::Vector4d(0, 0, 1, 1);
    case GREEN:
        return Eigen::Vector4d(0, 1, 0, 1);
    case PURPLE:
        return Eigen::Vector4d(0.3, 0, 0.6, 1);
    case YELLOW:
        return Eigen::Vector4d(1, 1, 0, 1);
    case GREY:
        return Eigen::Vector4d(0.6, 0.6, 0.6, 1);
    case PINK:
        return Eigen::Vector4d(1, 0, 0.5, 1);
    case ORANGE:
        return Eigen::Vector4d(1, 0.5, 0, 1);
    case AQUA:
        return Eigen::Vector4d(0, 1, 1, 1);
    default:
        return Eigen::Vector4d(0, 0, 0, 1);
    }
}

Eigen::Matrix4d get_clockwise_rotation(double angle_in_degrees)
{
    double angle_in_radians = 2 * M_PI * (angle_in_degrees / 360);

    // Rotating the object
    Eigen::Matrix4d rotation_transform;
    rotation_transform << cos(angle_in_radians), sin(angle_in_radians), 0, 0,
        -sin(angle_in_radians), cos(angle_in_radians), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return rotation_transform;
}

Eigen::Matrix4d get_anticlockwise_rotation(double angle_in_degrees)
{
    double angle_in_radians = 2 * M_PI * (angle_in_degrees / 360);

    // Rotating the object
    Eigen::Matrix4d rotation_transform;
    rotation_transform << cos(angle_in_radians), -sin(angle_in_radians), 0, 0,
        sin(angle_in_radians), cos(angle_in_radians), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return rotation_transform;
}

Eigen::Matrix4d get_translation(const Eigen::Vector3d &translation)
{
    Eigen::Matrix4d translation_matrix;
    translation_matrix << 1, 0, 0, translation(0),
        0, 1, 0, translation(1),
        0, 0, 1, translation(2),
        0, 0, 0, 1;
    return translation_matrix;
}

Eigen::Matrix4d get_scaling(double scale = 1)
{
    Eigen::Matrix4d scaling_transformation;
    scaling_transformation << scale, 0, 0, 0,
        0, scale, 0, 0,
        0, 0, scale, 0,
        0, 0, 0, 1;
    return scaling_transformation;
}

Eigen::Matrix4d get_camera_transformation(const Eigen::Vector3d &camera_position)
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

double ray_triangle_intersection(const Eigen::Vector3d &ray_origin, const Eigen::Vector3d &ray_direction, const Eigen::Vector3d &a, const Eigen::Vector3d &b, const Eigen::Vector3d &c)
{
    // Compute whether the ray intersects the given triangle.
    Eigen::Matrix3d M;
    M.col(0) << (a - b);
    M.col(1) << (a - c);
    M.col(2) << ray_direction;
    Eigen::Vector3d z = a - ray_origin;

    Eigen::Vector3d x = M.colPivHouseholderQr().solve(z);

    // A solution exists if t > 0, 0 <= u,v & u + v <= 1
    bool is_ray_intersect = x(0) >= 0 && x(1) >= 0 && (x(0) + x(1)) <= 1 && x(2) > 0;

    if (is_ray_intersect)
    {
        return x(2);
    }

    return -1;
}

// Finds the closest intersecting object returns its index
int find_nearest_object(const Eigen::Vector3d &ray_origin, const Eigen::Vector3d &ray_direction)
{
    int closest_index = -1;
    double closest_t = std::numeric_limits<double>::max(); // closest t is "+ infinity"

    for (int i = 0; i < triangle_vertices.size() / 3; i++)
    {
        VertexAttributes a = triangle_vertices[i * 3 + 0];
        VertexAttributes b = triangle_vertices[i * 3 + 1];
        VertexAttributes c = triangle_vertices[i * 3 + 2];

        Eigen::Vector4d vertex_a = model_transformations[i] * a.position;
        Eigen::Vector4d vertex_b = model_transformations[i] * b.position;
        Eigen::Vector4d vertex_c = model_transformations[i] * c.position;

        const double t = ray_triangle_intersection(
            ray_origin,
            ray_direction,
            vertex_a.head<3>(),
            vertex_b.head<3>(),
            vertex_c.head<3>());

        // We have intersection and the point is before our current closest t
        if (t >= 0 && t < closest_t)
        {
            closest_index = i;
            closest_t = t;
        }
    }

    return closest_index;
}

Eigen::Vector4d get_world_coordinates(int x_screen, int y_screen)
{
    Eigen::Vector4d canonical_coords = Eigen::Vector4d((double(x_screen) / double(width) * 2) - 1, (double(height - 1 - y_screen) / double(height) * 2) - 1, 0, 1);
    Eigen::Vector4d world_coords = inverse_transformation * canonical_coords;
    return world_coords;
}

VertexAttributes get_vertex_attributes(const Eigen::Vector4d &coordinates, color color_code = BLACK)
{
    VertexAttributes vertex(coordinates(0), coordinates(1), coordinates(2));
    vertex.color = get_color_vector(color_code);
    return vertex;
}

void highlight_triangle(int selected_index)
{
    if (selected_index > -1)
    {
        Eigen::Vector4d color = get_color_vector(BLUE);
        triangle_vertices[selected_index * 3 + 0].color = color;
        triangle_vertices[selected_index * 3 + 1].color = color;
        triangle_vertices[selected_index * 3 + 2].color = color;
    }
}

void remove_selection()
{
    if (selected_index > -1)
    {
        Eigen::Vector4d color = get_color_vector(RED);
        triangle_vertices[selected_index * 3 + 0].color = color;
        triangle_vertices[selected_index * 3 + 1].color = color;
        triangle_vertices[selected_index * 3 + 2].color = color;
    }
    selected_index = -1;
}

void insert_triangle(const Eigen::Vector4d &a, const Eigen::Vector4d &b, const Eigen::Vector4d &c)
{
    triangle_vertices.push_back(get_vertex_attributes(a, RED));
    triangle_vertices.push_back(get_vertex_attributes(b, RED));
    triangle_vertices.push_back(get_vertex_attributes(c, RED));

    model_transformations.push_back(Eigen::Matrix4d::Identity());
}

void insert_preview(const Eigen::Vector4d &world_coordinates)
{
    int num_line_vertices = line_vertices.size();

    // Add first vertex
    if (num_line_vertices == 0)
    {
        line_vertices.push_back(get_vertex_attributes(world_coordinates));
        line_vertices.push_back(get_vertex_attributes(world_coordinates));
    }
    // Add second vertex
    else if (num_line_vertices == 2)
    {
        line_vertices.back().position = world_coordinates;
        line_vertices.push_back(get_vertex_attributes(world_coordinates));
        line_vertices.push_back(get_vertex_attributes(world_coordinates));
        line_vertices.push_back(get_vertex_attributes(world_coordinates));
        line_vertices.push_back(line_vertices[0]);
    }
    else
    {
        // Add new triangle
        insert_triangle(line_vertices[0].position, line_vertices[1].position, world_coordinates);
        line_vertices.clear();
    }
}

void select_triangle(const Eigen::Vector4d &world_coordinates, const Eigen::Vector3d &camera_position)
{
    remove_selection();

    Eigen::Vector3d ray_origin = camera_position;
    Eigen::Vector3d ray_direction = (world_coordinates.head<3>() - ray_origin).normalized();

    // Select triangle
    selected_index = find_nearest_object(ray_origin, ray_direction);
    highlight_triangle(selected_index);
}

void delete_triangle(const Eigen::Vector4d &world_coordinates, const Eigen::Vector3d &camera_position)
{
    Eigen::Vector3d ray_origin = camera_position;
    Eigen::Vector3d ray_direction = (world_coordinates.head<3>() - camera_position).normalized();

    const int nearest_index = find_nearest_object(ray_origin, ray_direction);
    if (nearest_index > -1)
    {
        int index_to_remove = nearest_index * 3;
        triangle_vertices.erase(triangle_vertices.begin() + index_to_remove, triangle_vertices.begin() + index_to_remove + 3);
        model_transformations.erase(model_transformations.begin() + nearest_index);
    }
}

void update_preview(int x_screen, int y_screen)
{
    Eigen::Vector4d world_coords = get_world_coordinates(x_screen, y_screen);

    int num_vertices = line_vertices.size();
    if (num_vertices <= 2)
    {
        line_vertices.back().position = world_coords;
    }
    else
    {
        line_vertices[num_vertices - 2].position = world_coords;
        line_vertices[num_vertices - 3].position = world_coords;
    }
}

void update_translation(int xrel, int yrel)
{
    Eigen::Vector4d canonical_coords = Eigen::Vector4d((double(xrel) / double(width) * 2), (-double(yrel) / double(height) * 2), 0, 0);
    Eigen::Vector4d world_coords = inverse_transformation * canonical_coords;
    model_transformations[selected_index] = model_transformations[selected_index] * get_translation(world_coords.head<3>());
}

void update_rotation(bool is_clockwise = true)
{
    if (selected_index == -1)
    {
        return;
    }

    // Selected triangle
    Eigen::Vector4d a = triangle_vertices[selected_index * 3 + 0].position;
    Eigen::Vector4d b = triangle_vertices[selected_index * 3 + 1].position;
    Eigen::Vector4d c = triangle_vertices[selected_index * 3 + 2].position;

    // Compute barycenter of selected triangle
    double center_x = (a(0) + b(0) + c(0)) / 3.0;
    double center_y = (a(1) + b(1) + c(1)) / 3.0;
    double center_z = (a(2) + b(2) + c(2)) / 3.0;
    Eigen::Vector3d center_coords = Eigen::Vector3d(center_x, center_y, center_z);

    Eigen::Matrix4d rotation_matrix = is_clockwise ? get_clockwise_rotation(angle) : get_anticlockwise_rotation(angle);
    Eigen::Matrix4d rotation = get_translation(center_coords) * rotation_matrix * get_translation(-1 * center_coords);

    // Update rotation
    model_transformations[selected_index] = model_transformations[selected_index] * rotation;
}

void update_scaling(double scale_factor)
{
    if (selected_index == -1)
    {
        return;
    }

    // Selected triangle
    Eigen::Vector4d a = triangle_vertices[selected_index * 3 + 0].position;
    Eigen::Vector4d b = triangle_vertices[selected_index * 3 + 1].position;
    Eigen::Vector4d c = triangle_vertices[selected_index * 3 + 2].position;

    // Compute barycenter of selected triangle
    double center_x = (a(0) + b(0) + c(0)) / 3.0;
    double center_y = (a(1) + b(1) + c(1)) / 3.0;
    double center_z = (a(2) + b(2) + c(2)) / 3.0;
    Eigen::Vector3d center_coords = Eigen::Vector3d(center_x, center_y, center_z);

    Eigen::Matrix4d scale_transform = get_translation(center_coords) * get_scaling(scale_factor) * get_translation(-1 * center_coords);

    // Update rotation
    model_transformations[selected_index] = model_transformations[selected_index] * scale_transform;
}

void reset_previous_mode()
{
    if (current_mode != INSERT_MODE)
    {
        line_vertices.clear();
    }
    if (current_mode != TRANSLATE_MODE)
    {
        remove_selection();
    }
}

void change_mode(char key_pressed)
{
    switch (key_pressed)
    {
    case 'c':
        current_mode = COLOR_MODE;
        break;
    case 'i':
        current_mode = INSERT_MODE;
        break;
    case 'o':
        current_mode = TRANSLATE_MODE;
        break;
    case 'p':
        current_mode = DELETE_MODE;
        break;
    case 'z':
        current_mode = NONE;
        break;
    }
    reset_previous_mode();
}

void render_triangles_with_wireframe(
    const Program &program,
    UniformAttributes &uniform,
    const std::vector<VertexAttributes> triangle_vertices)
{
    Eigen::Vector4d black_color = get_color_vector(BLACK);
    std::vector<VertexAttributes> triangle;
    std::vector<VertexAttributes> lines;

    for (int i = 0; i < triangle_vertices.size() / 3; i++)
    {
        triangle.clear();

        VertexAttributes a = triangle_vertices[i * 3 + 0];
        VertexAttributes b = triangle_vertices[i * 3 + 1];
        VertexAttributes c = triangle_vertices[i * 3 + 2];

        triangle.push_back(a);
        triangle.push_back(b);
        triangle.push_back(c);

        // To avoid lines being hidden by faces
        a.position[2] += 0.0005;
        b.position[2] += 0.0005;
        c.position[2] += 0.0005;

        lines.clear();

        lines.push_back(a);
        lines.push_back(b);

        lines.push_back(b);
        lines.push_back(c);

        lines.push_back(c);
        lines.push_back(a);

        for (int i = 0; i < lines.size(); i++)
        {
            lines[i].color = black_color;
        }

        uniform.model_transformation = model_transformations[i];
        rasterize_triangles(program, uniform, triangle, frameBuffer);
        rasterize_lines(program, uniform, lines, 1, frameBuffer);
    }
}

int main(int argc, char *args[])
{
    // Global Constants (empty in this example)
    UniformAttributes uniform;

    // Basic rasterization program
    Program program;

    // The vertex shader is the identity
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform)
    {
        VertexAttributes out;
        out.position = uniform.world_transformation * uniform.model_transformation * va.position;
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
    uniform.world_transformation = view * ortho_projection * perspective_projection * camera_transformation;

    // Computing inverse transformation
    inverse_transformation = uniform.world_transformation.inverse();

    // Initialize the viewer and the corresponding callbacks
    SDLViewer viewer;
    viewer.init("Viewer Example", width, height);

    viewer.mouse_move = [&](int x, int y, int xrel, int yrel)
    {
        switch (current_mode)
        {
        case INSERT_MODE:
            if (line_vertices.size())
            {
                update_preview(x, y);
                viewer.redraw_next = true;
            }
            break;
        case TRANSLATE_MODE:
            if (is_mouse_pressed && selected_index > -1)
            {
                update_translation(xrel, yrel);
                viewer.redraw_next = true;
            }
        default:
            break;
        }
    };

    viewer.mouse_pressed = [&](int x, int y, bool is_pressed, int button, int clicks)
    {
        Eigen::Vector4d world_coords = get_world_coordinates(x, y);
        is_mouse_pressed = is_pressed;

        if (!is_mouse_pressed)
        {
            return;
        }

        switch (current_mode)
        {
        case INSERT_MODE:
            insert_preview(world_coords);
            break;
        case TRANSLATE_MODE:
            select_triangle(world_coords, uniform.camera_position);
            break;
        case DELETE_MODE:
            delete_triangle(world_coords, uniform.camera_position);
            break;
        default:
            break;
        }

        viewer.redraw_next = true;
    };

    viewer.mouse_wheel = [&](int dx, int dy, bool is_direction_normal) {
    };

    viewer.key_pressed = [&](char key, bool is_pressed, int modifier, int repeat)
    {
        switch (key)
        {
        case 'c':
        case 'i':
        case 'o':
        case 'p':
        case 'z':
            change_mode(key);
            break;
        case 'h':
            update_rotation();
            break;
        case 'j':
            update_rotation(false);
            break;
        case 'k':
            update_scaling(1.25);
            break;
        case 'l':
            update_scaling(0.75);
            break;
        default:
            break;
        }
        viewer.redraw_next = true;
    };

    viewer.redraw = [&](SDLViewer &viewer)
    {
        // Clear the framebuffer
        frameBuffer.setConstant(FrameBufferAttributes());

        // Render inserted triangles
        render_triangles_with_wireframe(program, uniform, triangle_vertices);

        // Render insert preview
        uniform.model_transformation.setIdentity();
        rasterize_lines(program, uniform, line_vertices, 1, frameBuffer);

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
