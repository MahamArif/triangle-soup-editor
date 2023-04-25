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
const double aspect_ratio = double(frameBuffer.cols()) / double(frameBuffer.rows());
const double near = 0.1;
const double far = 100;

// For zoom in/zoom out
double zoom_factor = 1.0;

// To pan the view
double x_offset = 0.0;
double y_offset = 0.0;

// For highlighting triangle
int selected_triangle = -1;
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

// To change the color of vertex
int selected_vertex = -1;

// Modes
enum Mode
{
    NONE = 0,
    INSERT_MODE = 1,
    TRANSLATE_MODE = 2,
    DELETE_MODE = 3,
    COLOR_MODE = 4
};

Mode current_mode;

// Colors
enum Color
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

Eigen::Vector4d get_color_vector(Color color_code)
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

Eigen::Matrix4d lerp_matrices(const Eigen::Matrix4d &mat1, const Eigen::Matrix4d &mat2, double interpolation_factor, const Eigen::Vector3d &barycenter) {
    // Convert into affine transformation
    Eigen::Affine3d affine1 = Eigen::Transform<double, 3, Eigen::Affine>(mat1);
    Eigen::Affine3d affine2 = Eigen::Transform<double, 3, Eigen::Affine>(mat2);

    // Decompose input transformations into translation, rotation, and scaling components
    // Translation
    Eigen::Translation3d translation1(affine1.translation());
    Eigen::Translation3d translation2(affine2.translation());

    // Rotation
    Eigen::Quaterniond rotation1(affine1.rotation());
    Eigen::Quaterniond rotation2(affine2.rotation());

    // Scaling
    double sx1 = mat1.block<3, 1>(0, 0).norm();
    double sy1 = mat1.block<3, 1>(0, 1).norm();
    double sz1 = mat1.block<3, 1>(0, 2).norm();
    Eigen::Vector3d scaling1 = Eigen::Vector3d(sx1, sy1, sz1);

    double sx2 = mat2.block<3, 1>(0, 0).norm();
    double sy2 = mat2.block<3, 1>(0, 1).norm();
    double sz2 = mat2.block<3, 1>(0, 2).norm();
    Eigen::Vector3d scaling2 = Eigen::Vector3d(sx2, sy2, sz2);

    // Interpolate translations using lerp
    Eigen::Vector3d interpolated_translation = translation1.vector() + interpolation_factor * (translation2.vector() - translation1.vector());

    // Interpolate rotations using slerp
    Eigen::Quaterniond interpolated_rotation = rotation1.slerp(interpolation_factor, rotation2);

    // Interpolate scalings using lerp
    Eigen::Vector3d interpolated_scaling = scaling1 + interpolation_factor * (scaling2 - scaling1);

    // Create the transformation matrix
    Eigen::Affine3d interpolated_transform = Eigen::Affine3d::Identity();

    // Translate the object so that its barycenter is at the origin
    interpolated_transform.translate(-barycenter);

    // Apply the interpolated rotation and scaling transformations
    interpolated_transform.rotate(interpolated_rotation);
    interpolated_transform.scale(interpolated_scaling);

    // Translate the object back to its original position
    interpolated_transform.translate(barycenter);

    // Apply the interpolated translation
    interpolated_transform.translate(interpolated_translation);

    return interpolated_transform.matrix();
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

Eigen::Matrix4d get_orthographic_projection()
{
    // Specifying the bounding box coordinates for canonical cube
    double left = (-1 + x_offset) * zoom_factor;
    double right = (1 + x_offset) * zoom_factor;
    double bottom = (-1 + y_offset) * zoom_factor;
    double top = (1 + y_offset) * zoom_factor;

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

VertexAttributes get_vertex_attributes(const Eigen::Vector4d &coordinates, Color color_code = BLACK)
{
    VertexAttributes vertex(coordinates(0), coordinates(1), coordinates(2));
    vertex.color = get_color_vector(color_code);
    return vertex;
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
    Eigen::Vector3d ray_origin = camera_position;
    Eigen::Vector3d ray_direction = (world_coordinates.head<3>() - ray_origin).normalized();

    // Select triangle
    selected_triangle = find_nearest_object(ray_origin, ray_direction);
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
    model_transformations[selected_triangle] = get_translation(world_coords.head<3>()) * model_transformations[selected_triangle];
}

void update_rotation(bool is_clockwise = true)
{
    if (selected_triangle == -1)
    {
        return;
    }

    // Selected triangle
    Eigen::Vector4d a = triangle_vertices[selected_triangle * 3 + 0].position;
    Eigen::Vector4d b = triangle_vertices[selected_triangle * 3 + 1].position;
    Eigen::Vector4d c = triangle_vertices[selected_triangle * 3 + 2].position;

    // Compute barycenter of selected triangle
    double center_x = (a(0) + b(0) + c(0)) / 3.0;
    double center_y = (a(1) + b(1) + c(1)) / 3.0;
    double center_z = (a(2) + b(2) + c(2)) / 3.0;
    Eigen::Vector3d center_coords = Eigen::Vector3d(center_x, center_y, center_z);

    Eigen::Matrix4d rotation_matrix = is_clockwise ? get_clockwise_rotation(angle) : get_anticlockwise_rotation(angle);
    Eigen::Matrix4d rotation = get_translation(center_coords) * rotation_matrix * get_translation(-1 * center_coords);

    // Update rotation
    model_transformations[selected_triangle] = model_transformations[selected_triangle] * rotation;
}

void update_scaling(double scale_factor)
{
    if (selected_triangle == -1)
    {
        return;
    }

    // Selected triangle
    Eigen::Vector4d a = triangle_vertices[selected_triangle * 3 + 0].position;
    Eigen::Vector4d b = triangle_vertices[selected_triangle * 3 + 1].position;
    Eigen::Vector4d c = triangle_vertices[selected_triangle * 3 + 2].position;

    // Compute barycenter of selected triangle
    double center_x = (a(0) + b(0) + c(0)) / 3.0;
    double center_y = (a(1) + b(1) + c(1)) / 3.0;
    double center_z = (a(2) + b(2) + c(2)) / 3.0;
    Eigen::Vector3d center_coords = Eigen::Vector3d(center_x, center_y, center_z);

    Eigen::Matrix4d scale_transform = get_translation(center_coords) * get_scaling(scale_factor) * get_translation(-1 * center_coords);

    // Update rotation
    model_transformations[selected_triangle] = model_transformations[selected_triangle] * scale_transform;
}

void update_transformation(UniformAttributes &uniform)
{
    Eigen::Matrix4d camera_transformation = get_camera_transformation(uniform.camera_position);
    Eigen::Matrix4d perspective_projection = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d ortho_projection = get_orthographic_projection();
    Eigen::Matrix4d view = get_view_transformation(aspect_ratio);
    uniform.world_transformation = view * ortho_projection * perspective_projection * camera_transformation;

    // Computing inverse transformation
    inverse_transformation = uniform.world_transformation.inverse();
}

// Finds the closest vertex to the mouse position
int find_closest_vertex(const Eigen::Vector4d &mouse_position)
{
    int closest_index = -1;
    double closest_distance = std::numeric_limits<double>::max(); // closest distance is "+ infinity"

    for (int i = 0; i < triangle_vertices.size(); i++)
    {
        Eigen::Vector4d vertex = model_transformations[i / 3] * triangle_vertices[i].position;
        const double distance = (vertex - mouse_position).norm();

        if (distance < closest_distance)
        {
            closest_index = i;
            closest_distance = distance;
        }
    }

    return closest_index;
}

void change_vertex_color(char key)
{
    if (current_mode == COLOR_MODE && selected_vertex > -1)
    {
        int color_code = key - '0';
        Color selected_color = static_cast<Color>(color_code);
        triangle_vertices[selected_vertex].color = get_color_vector(selected_color);
    }
}

void reset_previous_mode()
{
    if (current_mode != INSERT_MODE)
    {
        line_vertices.clear();
    }
    if (current_mode != TRANSLATE_MODE)
    {
        selected_triangle = -1;
    }
    if (current_mode != COLOR_MODE)
    {
        selected_vertex = -1;
    }
}

void change_mode(char key_pressed)
{
    switch (key_pressed)
    {
    case SDLK_c:
        current_mode = COLOR_MODE;
        break;
    case SDLK_i:
        current_mode = INSERT_MODE;
        break;
    case SDLK_o:
        current_mode = TRANSLATE_MODE;
        break;
    case SDLK_p:
        current_mode = DELETE_MODE;
        break;
    case SDLK_ESCAPE:
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

        // To avoid lines being hidden by triangle
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
        uniform.is_object_highlighted = selected_triangle == i;
        rasterize_triangles(program, uniform, triangle, frameBuffer);

        uniform.is_object_highlighted = false;
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
        out.color = uniform.is_object_highlighted ? get_color_vector(BLUE) : va.color;
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

    // Orthographic Projection
    // Set the transformations for camera space and orthographic projection
    uniform.camera_position = Eigen::Vector3d(0, 0, -5);

    update_transformation(uniform);

    // Initialize the viewer and the corresponding callbacks
    SDLViewer viewer;
    viewer.init("Triangle Soup Editor", width, height);

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
            if (is_mouse_pressed && selected_triangle > -1)
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
        case COLOR_MODE:
            selected_vertex = find_closest_vertex(world_coords);
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
        if (!is_pressed)
        {
            return;
        }

        switch (key)
        {
        case SDLK_c:
        case SDLK_i:
        case SDLK_o:
        case SDLK_p:
        case SDLK_ESCAPE:
            change_mode(key);
            break;
        case SDLK_h:
            update_rotation();
            break;
        case SDLK_j:
            update_rotation(false);
            break;
        case SDLK_k:
            update_scaling(1.25);
            break;
        case SDLK_l:
            update_scaling(0.75);
            break;
        case SDLK_1:
        case SDLK_2:
        case SDLK_3:
        case SDLK_4:
        case SDLK_5:
        case SDLK_6:
        case SDLK_7:
        case SDLK_8:
        case SDLK_9:
            change_vertex_color(key);
            break;
        case SDLK_PLUS:
        case SDLK_KP_PLUS:
        case SDLK_EQUALS:       // '+' might be detected as '=' (shift + '=')
            zoom_factor /= 1.2; // Decrease zoom factor to zoom in
            update_transformation(uniform);
            break;
        case SDLK_MINUS:
        case SDLK_KP_MINUS:
            zoom_factor *= 1.2; // Increase zoom factor to zoom out
            update_transformation(uniform);
            break;
        case SDLK_w:
            y_offset += 0.2;
            update_transformation(uniform);
            break;
        case SDLK_a:
            x_offset -= 0.2;
            update_transformation(uniform);
            break;
        case SDLK_s:
            y_offset -= 0.2;
            update_transformation(uniform);
            break;
        case SDLK_d:
            x_offset += 0.2;
            update_transformation(uniform);
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
        uniform.is_object_highlighted = false;
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
