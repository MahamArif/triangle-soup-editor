#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

class VertexAttributes
{
public:
	VertexAttributes(double x = 0, double y = 0, double z = 0, double w = 1)
	{
		position << x, y, z, w;
		color << 1, 1, 1, 1;
		object_id = -1;
	}

	// Interpolates the vertex attributes
	static VertexAttributes interpolate(
		const VertexAttributes &a,
		const VertexAttributes &b,
		const VertexAttributes &c,
		const double alpha,
		const double beta,
		const double gamma)
	{
		VertexAttributes r;
		r.position = alpha * (a.position / a.position[3]) + beta * (b.position / b.position[3]) + gamma * (c.position / c.position[3]);
		r.color = alpha * a.color + beta * b.color + gamma * c.color;
		return r;
	}

	Eigen::Vector4d position;
	Eigen::Vector4d color;
	int object_id;
};

class FragmentAttributes
{
public:
	FragmentAttributes(double r = 0, double g = 0, double b = 0, double a = 1)
	{
		color << r, g, b, a;
	}

	Eigen::Vector4d color;
	Eigen::Vector4d position;
};

class FrameBufferAttributes
{
public:
	FrameBufferAttributes(uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255)
	{
		color << r, g, b, a;
		depth = 2; // this value should be between -1 and 1, 2 is further than the visible range
	}

	Eigen::Matrix<uint8_t, 4, 1> color;
	double depth;
};

class UniformAttributes
{
public:
	Eigen::Matrix4d transformation;
	Eigen::Matrix4d inverse_transformation;
	Eigen::Vector3d camera_position;
	std::vector<Eigen::Matrix4d> model_transformations;
};