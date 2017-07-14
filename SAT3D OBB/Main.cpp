/*
Title: SAT-3D (OBB)
File Name: Main.cpp
Copyright © 2015
Original authors: Brockton Roth
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:
This is a Separating Axis Theorem test. (Sometimes just called Separating Axis Test.) This is in 3D.
Contains two cubes, one that is stationary and one that is moving. They are bounded by OBBs
(Object-Oriented Bounding Boxes) and when these OBBs collide the moving object "bounces" on the x
axis (because that is the only direction the object is moving). The algorithm will detect any axis
of collision, but will not output the axis that was collided (because it doesn't know). Thus, we
assume x and hardcode in the x axis bounce.
There is a physics timestep such that every update runs at the same delta time, regardless of how
fast or slow the computer is running. The cubes would be the exact same as their OBBs, since they are
aligned on the same axis.
*/

#include "GLIncludes.h"
#include "GameObject.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// This is your reference to your shader program.
// This will be assigned with glCreateProgram().
// This program will run on your GPU.
GLuint program;

// These are your references to your actual compiled shaders
GLuint vertex_shader;
GLuint fragment_shader;

// This is a reference to your uniform MVP matrix in your vertex shader
GLuint uniMVP;

// These are 4x4 transformation matrices, which you will locally modify before passing into the vertex shader via uniMVP
glm::mat4 proj;
glm::mat4 view;

// proj * view = PV
glm::mat4 PV;

// MVP is PV * Model (model is the transformation matrix of whatever object is being rendered)
glm::mat4 MVP;
glm::mat4 MVP2;

// Variables for FPS and Physics Timestep calculations.
int frame = 0;
double time = 0;
double timebase = 0;
double accumulator = 0.0;
int fps = 0;
double FPSTime = 0.0;
double physicsStep = 0.012; // This is the number of milliseconds we intend for the physics to update.

// Variable for the speed of the moving object.
float speed = 0.90f;

bool antiStuck = false;

// Reference to the window object being created by GLFW.
GLFWwindow* window;

// An array of vertices stored in an std::vector for our object.
std::vector<VertexFormat> vertices;

// References to our two GameObjects and the one Model we'll be using.
GameObject* obj1;
GameObject* obj2;
Model* cube;

struct OBB
{
	glm::vec3 corners[8];
};

OBB obb1;
OBB obb2;

// Returns the normals for an OBB's faces.
std::vector<glm::vec3> GetNormals(OBB obj)
{
	// Since the OBB is a cube, there should be 6 normals
	// But it's a cube, which means that every opposite side will have the same normal just in the reverse direction.
	// SAT doesn't care about the direction of the normal, only the axis, so we can ignore half the normals.
	std::vector<glm::vec3> normals;

	// Below is an implementation for calculating the normals of a given cube/OBB, assuming the points are ordered in groups of 3 where each 3 points is 
	// a triangle on a face.

	// 3 cube normals for the faces
	// 0, 1, 2 FRONT
	// 3, 2, 4 RIGHT
	// 1, 6, 4 TOP

	glm::vec3 U(obj.corners[1] - obj.corners[0]);
	glm::vec3 V(obj.corners[2] - obj.corners[0]);
	normals.push_back(glm::normalize(glm::vec3((U.y * V.z) - (U.z * V.y), (U.z * V.x) - (U.x * V.z), (U.x * V.y) - (U.y * V.x))));

	U = glm::vec3(obj.corners[2] - obj.corners[3]);
	V = glm::vec3(obj.corners[4] - obj.corners[3]);
	normals.push_back(glm::normalize(glm::vec3((U.y * V.z) - (U.z * V.y), (U.z * V.x) - (U.x * V.z), (U.x * V.y) - (U.y * V.x))));

	U = glm::vec3(obj.corners[6] - obj.corners[1]);
	V = glm::vec3(obj.corners[4] - obj.corners[1]);
	normals.push_back(glm::normalize(glm::vec3((U.y * V.z) - (U.z * V.y), (U.z * V.x) - (U.x * V.z), (U.x * V.y) - (U.y * V.x))));

	return normals;
}

// Gets the normals (cross products) of the 9 possible pairs of edges between the two OBBs.
std::vector<glm::vec3> GetEdgeNormals(OBB first, OBB second)
{
	// In OBB Collision, we must get the 9 cross products of the edge combinations.
	// There are 9 because there are 3 unique edges on each cube, which otherwise have parallel edges on other parts of the cube.
	// Each of these 3 edges is then used in the cross product with 3 edges from the other cube, resulting in 9 total possible combinations.
	
	// 3 unique cube edges
	// 0, 1
	// 1, 2
	// 1, 6
	
	std::vector<glm::vec3> normals;

	glm::vec3 firstEdges[3];
	glm::vec3 secondEdges[3];

	firstEdges[0] = first.corners[1] - first.corners[0];
	firstEdges[1] = first.corners[2] - first.corners[1];
	firstEdges[2] = first.corners[6] - first.corners[1];

	secondEdges[0] = second.corners[1] - second.corners[0];
	secondEdges[1] = second.corners[2] - second.corners[1];
	secondEdges[2] = second.corners[6] - second.corners[1];

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			normals.push_back(glm::normalize(glm::cross(firstEdges[i], secondEdges[j])));
		}
	}

	return normals;
}

// Gets the minimum and maximum projections for a set of points given an axis to project them onto.
// Will output the values into the min and max variables passed into the function.
void GetMinMax(OBB obj, glm::vec3 axis, float& min, float&max)
{
	// To get a projection along a vector, you take the dot product of the vertex (in vector form) with the axis vector.
	// Since we're looking for both a minimum and a maximum, we start with the first point, and go from there.
	min = glm::dot(obj.corners[0], axis);
	max = min;

	// Looping through the rest of the points.
	for (int i = 1; i < 8; i++)
	{
		// Find the projection for the current point.
		float currProj = glm::dot(obj.corners[i], axis);

		// If this projection is smaller than our minimum projection, the minimum becomes this.
		if (min > currProj)
		{
			min = currProj;
		}

		// If this projection is larger than our maximum projection, the maximum becomes this.
		if (currProj > max)
		{
			max = currProj;
		}
	}
}

// Returns true if there is no collision, and false if there is a collision.
bool TestSAT(OBB a, OBB b)
{
	// The first step is to get the normals for the two colliding objects, as these will be the axes on which we test for collisions.
	std::vector<glm::vec3> aNormals = GetNormals(a);
	std::vector<glm::vec3> bNormals = GetNormals(b);
	std::vector<glm::vec3> edgeNormals = GetEdgeNormals(a, b);

	// This boolean gets returned, and will be true if there is no collision.
	bool isSeparated = false;

	// For each normal
	for (int i = 0; i < aNormals.size(); i++)
	{
		// Get the Min and Max projections for each object along the normal.
		float aMin, aMax;
		GetMinMax(a, aNormals[i], aMin, aMax);

		float bMin, bMax;
		GetMinMax(b, aNormals[i], bMin, bMax);

		// If the maximum projection of one of the objects is less than the minimum projection of the other object, then we can determine that there is a separation 
		// along this axis. Thus, we set isSeparated to true and break out of the for loop.
		isSeparated = aMax < bMin || bMax < aMin;
		if (isSeparated) break;
	}

	// This only runs if we still haven't proven that there is a separation between the two objects.
	// SAT is an optimistic algorithm in that it will stop the moment it determines there isn't a collision, and as such the less collisions there are the faster it will run.
	if (!isSeparated)
	{
		// Loop through the normals for the second object.
		// The process below is exactly the same as above, only with object b's normals instead of object a.
		for (int i = 0; i < bNormals.size(); i++)
		{
			float aMin, aMax;
			GetMinMax(a, bNormals[i], aMin, aMax);

			float bMin, bMax;
			GetMinMax(b, bNormals[i], bMin, bMax);

			isSeparated = aMax < bMin || bMax < aMin;
			if (isSeparated) break;
		}
	}

	// This only runs if we still haven't proven that there is a separation between the two objects.
	// SAT is an optimistic algorithm in that it will stop the moment it determines there isn't a collision, and as such the less collisions there are the faster it will run.
	if (!isSeparated)
	{
		// Loop through the normals for the edges.
		// The process below is exactly the same as above, only with the edge normals instead of face normals.
		for (int i = 0; i < edgeNormals.size(); i++)
		{
			float aMin, aMax;
			GetMinMax(a, edgeNormals[i], aMin, aMax);

			float bMin, bMax;
			GetMinMax(b, edgeNormals[i], bMin, bMax);

			isSeparated = aMax < bMin || bMax < aMin;
			if (isSeparated) break;
		}
	}

	// At this point, isSeparated has been tested against each normal. If it has been set to true, then there is a separation. If it is false, that means none of the axes 
	// were separated, and there is a collision.
	return isSeparated;
}

// This runs once every physics timestep.
void update(float dt)
{
#pragma region Boundaries
	// This section just checks to make sure the object stays within a certain boundary. This is not really collision detection.
	glm::vec3 tempPos = obj2->GetPosition();
	
	if (fabsf(tempPos.x) > 1.35f)
	{
		glm::vec3 tempVel = obj2->GetVelocity();

		// "Bounce" the velocity along the axis that was over-extended.
		obj2->SetVelocity(glm::vec3(-1.0f * tempVel.x, tempVel.y, tempVel.z));
	}
	if (fabsf(tempPos.y) > 0.8f)
	{
		glm::vec3 tempVel = obj2->GetVelocity();
		obj2->SetVelocity(glm::vec3(tempVel.x, -1.0f * tempVel.y, tempVel.z));
	}
	if (fabsf(tempPos.z) > 1.0f)
	{
		glm::vec3 tempVel = obj2->GetVelocity();
		obj2->SetVelocity(glm::vec3(tempVel.x, tempVel.y, -1.0f * tempVel.z));
	}
#pragma endregion Boundaries section just bounces the object so it does not fly off the side of the screen infinitely.

	// Rotate the objects. This helps illustrate how the OBB follows the object's orientation.
	 obj1->Rotate(glm::vec3(glm::radians(1.0f), glm::radians(1.0f), glm::radians(0.0f)));
	 obj2->Rotate(glm::vec3(glm::radians(1.0f), glm::radians(1.0f), glm::radians(0.0f)));

	// Re-calculate the Object-Oriented Bounding Box for your object.
	// We do this because if the object's orientation changes, we should update the bounding box as well.
	// Be warned: For some objects this can actually cause a collision to be missed, so be careful.
	// (This is because we determine the collision based on the OBB, but if the OBB changes significantly, the time of collision can change between frames,
	// and if that lines up just right you'll miss the collision altogether.)
	 glm::vec4 pointsA[8];
	 glm::vec4 pointsB[8];

	 for (int i = 0; i < 8; i++)
	 {
		 pointsA[i] = *obj1->GetTransform() * glm::vec4(obj1->GetModel()->Vertices()[i].position, 1.0f);
		 pointsB[i] = *obj2->GetTransform() * glm::vec4(obj2->GetModel()->Vertices()[i].position, 1.0f);

		 obb1.corners[i] = glm::vec3(pointsA[i].x, pointsA[i].y, pointsA[i].z);
		 obb2.corners[i] = glm::vec3(pointsB[i].x, pointsB[i].y, pointsB[i].z);
	 }

	// Pass in our two objects to the SAT test, if it returns false then they are colliding because there is no separating axis between them.
	if (!TestSAT(obb2, obb1) && !antiStuck)
	{
		glm::vec3 velocity = obj2->GetVelocity();
		
		// Reverse the velocity in the x direction
		// This is the "bounce" effect, only we don't actually know the axis of collision from the test. Instead, we assume it because the object is only moving in the x 
		// direction.
		velocity.x *= -1;

		obj2->SetVelocity(velocity);

		// This variable exists to allow an extra frame/update to pass when they collide so that the objects do not get stuck as a result of tunneling and 
		// recalculating the OBB.
		antiStuck = true;
	}
	else if (antiStuck)
	{
		antiStuck = false;
	}
	
	// Update the objects based on their velocities.
	obj1->Update(dt);
	obj2->Update(dt);

	// Update your MVP matrices based on the objects' transforms.
	MVP = PV * *obj1->GetTransform();
	MVP2 = PV * *obj2->GetTransform();
}

// This runs once every frame to determine the FPS and how often to call update based on the physics step.
void checkTime()
{
	// Get the current time.
	time = glfwGetTime();

	// Get the time since we last ran an update.
	double dt = time - timebase;

	// If more time has passed than our physics timestep.
	if (dt > physicsStep)
	{
		// Calculate FPS: Take the number of frames (frame) since the last time we calculated FPS, and divide by the amount of time that has passed since the 
		// last time we calculated FPS (time - FPSTime).
		if (time - FPSTime > 1.0)
		{
			fps = frame / (time - FPSTime);

			FPSTime = time; // Now we set FPSTime = time, so that we have a reference for when we calculated the FPS
			
			frame = 0; // Reset our frame counter to 0, to mark that 0 frames have passed since we calculated FPS (since we literally just did it)

			std::string s = "FPS: " + std::to_string(fps); // This just creates a string that looks like "FPS: 60" or however much.

			glfwSetWindowTitle(window, s.c_str()); // This will set the window title to that string, displaying the FPS as the window title.
		}

		timebase = time; // Set timebase = time so we have a reference for when we ran the last physics timestep.

		// Limit dt so that we if we experience any sort of delay in processing power or the window is resizing/moving or anything, it doesn't update a bunch of times while the player can't see.
		// This will limit it to a .25 seconds.
		if (dt > 0.25)
		{
			dt = 0.25;
		}

		// The accumulator is here so that we can track the amount of time that needs to be updated based on dt, but not actually update at dt intervals and instead use our physicsStep.
		accumulator += dt;

		// Run a while loop, that runs update(physicsStep) until the accumulator no longer has any time left in it (or the time left is less than physicsStep, at which point it save that 
		// leftover time and use it in the next checkTime() call.
		while (accumulator >= physicsStep)
		{
			update(physicsStep);

			accumulator -= physicsStep;
		}
	}
}

// This function runs every frame
void renderScene()
{
	// Clear the color buffer and the depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Clear the screen to white
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// Tell OpenGL to use the shader program you've created.
	glUseProgram(program);

	// Set the uniform matrix in our shader to our MVP matrix for the first object.
	glUniformMatrix4fv(uniMVP, 1, GL_FALSE, glm::value_ptr(MVP));

	// Draw the cube.
	cube->Draw();

	// Set the uniform matrix in our shader to our MVP matrix for the second object.
	glUniformMatrix4fv(uniMVP, 1, GL_FALSE, glm::value_ptr(MVP2));

	// Draw the cube again.
	cube->Draw();

	// We're using the same model here to draw, but different transformation matrices so that we can use less data overall.
	// This is a technique called instancing, although "true" instancing involves binding a matrix array to the uniform variable and using DrawInstanced in place of draw.
}

// This method reads the text from a file.
// Realistically, we wouldn't want plain text shaders hardcoded in, we'd rather read them in from a separate file so that the shader code is separated.
std::string readShader(std::string fileName)
{
	std::string shaderCode;
	std::string line;

	// We choose ifstream and std::ios::in because we are opening the file for input into our program.
	// If we were writing to the file, we would use ofstream and std::ios::out.
	std::ifstream file(fileName, std::ios::in);

	// This checks to make sure that we didn't encounter any errors when getting the file.
	if (!file.good())
	{
		std::cout << "Can't read file: " << fileName.data() << std::endl;

		// Return so we don't error out.
		return "";
	}

	// ifstream keeps an internal "get" position determining the location of the element to be read next
	// seekg allows you to modify this location, and tellg allows you to get this location
	// This location is stored as a streampos member type, and the parameters passed in must be of this type as well
	// seekg parameters are (offset, direction) or you can just use an absolute (position).
	// The offset parameter is of the type streamoff, and the direction is of the type seekdir (an enum which can be ios::beg, ios::cur, or ios::end referring to the beginning, 
	// current position, or end of the stream).
	file.seekg(0, std::ios::end);					// Moves the "get" position to the end of the file.
	shaderCode.resize((unsigned int)file.tellg());	// Resizes the shaderCode string to the size of the file being read, given that tellg will give the current "get" which is at the end of the file.
	file.seekg(0, std::ios::beg);					// Moves the "get" position to the start of the file.

	// File streams contain two member functions for reading and writing binary data (read, write). The read function belongs to ifstream, and the write function belongs to ofstream.
	// The parameters are (memoryBlock, size) where memoryBlock is of type char* and represents the address of an array of bytes are to be read from/written to.
	// The size parameter is an integer that determines the number of characters to be read/written from/to the memory block.
	file.read(&shaderCode[0], shaderCode.size());	// Reads from the file (starting at the "get" position which is currently at the start of the file) and writes that data to the beginning
	// of the shaderCode variable, up until the full size of shaderCode. This is done with binary data, which is why we must ensure that the sizes are all correct.

	file.close(); // Now that we're done, close the file and return the shaderCode.

	return shaderCode;
}

// This method will consolidate some of the shader code we've written to return a GLuint to the compiled shader.
// It only requires the shader source code and the shader type.
GLuint createShader(std::string sourceCode, GLenum shaderType)
{
	// glCreateShader, creates a shader given a type (such as GL_VERTEX_SHADER) and returns a GLuint reference to that shader.
	GLuint shader = glCreateShader(shaderType);
	const char *shader_code_ptr = sourceCode.c_str(); // We establish a pointer to our shader code string
	const int shader_code_size = sourceCode.size();   // And we get the size of that string.

	// glShaderSource replaces the source code in a shader object
	// It takes the reference to the shader (a GLuint), a count of the number of elements in the string array (in case you're passing in multiple strings), a pointer to the string array 
	// that contains your source code, and a size variable determining the length of the array.
	glShaderSource(shader, 1, &shader_code_ptr, &shader_code_size);
	glCompileShader(shader); // This just compiles the shader, given the source code.

	GLint isCompiled = 0;

	// Check the compile status to see if the shader compiled correctly.
	glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);

	if (isCompiled == GL_FALSE)
	{
		char infolog[1024];
		glGetShaderInfoLog(shader, 1024, NULL, infolog);

		// Print the compile error.
		std::cout << "The shader failed to compile with the error:" << std::endl << infolog << std::endl;

		// Provide the infolog in whatever manor you deem best.
		// Exit with failure.
		glDeleteShader(shader); // Don't leak the shader.

		// NOTE: I almost always put a break point here, so that instead of the program continuing with a deleted/failed shader, it stops and gives me a chance to look at what may 
		// have gone wrong. You can check the console output to see what the error was, and usually that will point you in the right direction.
	}

	return shader;
}

// Initialization code
void init()
{	
	// Initializes the glew library
	glewInit();

	// Enables the depth test, which you will want in most cases. You can disable this in the render loop if you need to.
	glEnable(GL_DEPTH_TEST);

	// An element array, which determines which of the vertices to display in what order. This is sometimes known as an index array.
	GLuint elements[] = {
		0, 1, 2, 0, 2, 3, 3, 2, 4, 3, 4, 5, 5, 4, 6, 5, 6, 7, 7, 6, 1, 7, 1, 0, 1, 6, 4, 1, 4, 2, 7, 0, 3, 7, 3, 5
	};
	// These are the indices for a cube.
	vertices.push_back(VertexFormat(glm::vec3(-0.25, -0.25, 0.25),		// Front, Bottom, Left		0
		glm::vec4(1.0, 0.0, 0.0, 1.0))); //red
	vertices.push_back(VertexFormat(glm::vec3(-0.25, 0.25, 0.25),		// Front, Top, Left			1
		glm::vec4(1.0, 0.0, 0.0, 1.0))); //red
	vertices.push_back(VertexFormat(glm::vec3(0.25, 0.25, 0.25),		// Front, Top, Right		2
		glm::vec4(1.0, 0.0, 1.0, 1.0))); //yellow
	vertices.push_back(VertexFormat(glm::vec3(0.25, -0.25, 0.25),		// Front, Bottom, Right		3
		glm::vec4(1.0, 0.0, 1.0, 1.0))); //yellow
	vertices.push_back(VertexFormat(glm::vec3(0.25, 0.25, -0.25),		// Back, Top, Right			4
		glm::vec4(0.0, 1.0, 1.0, 1.0))); //cyan
	vertices.push_back(VertexFormat(glm::vec3(0.25, -0.25, -0.25),		// Back, Bottom, Right		5
		glm::vec4(0.0, 1.0, 1.0, 1.0))); //cyan
	vertices.push_back(VertexFormat(glm::vec3(-0.25, 0.25, -0.25),		// Back, Top, Left			6
		glm::vec4(0.0, 1.0, 0.0, 1.0))); //blue
	vertices.push_back(VertexFormat(glm::vec3(-0.25, -0.25, -0.25),		// Back, Bottom, Left		7
		glm::vec4(0.0, 1.0, 0.0, 1.0))); //blue

	// Create our icosphere model from the calculated data.
	cube = new Model(vertices.size(), vertices.data(), 36, elements);

	// Create two GameObjects based off of the icosphere model (note that they are both holding pointers to the icosphere, not actual copies of the icosphere vertex data).
	obj1 = new GameObject(cube);
	obj2 = new GameObject(cube);

	// Set beginning properties of GameObjects.
	obj1->SetVelocity(glm::vec3(0, 0.0f, 0.0f)); // The first object doesn't move.
	obj2->SetVelocity(glm::vec3(-speed, 0.0f, 0.0f));
	obj1->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	obj2->SetPosition(glm::vec3(-0.7f, 0.0f, 0.0f));
	obj1->SetScale(glm::vec3(0.85f, 0.85f, 0.85f));
	obj2->SetScale(glm::vec3(0.20f, 0.20f, 0.20f));

	// Read in the shader code from a file.
	std::string vertShader = readShader("../VertexShader.glsl");
	std::string fragShader = readShader("../FragmentShader.glsl");

	// createShader consolidates all of the shader compilation code
	vertex_shader = createShader(vertShader, GL_VERTEX_SHADER);
	fragment_shader = createShader(fragShader, GL_FRAGMENT_SHADER);

	// A shader is a program that runs on your GPU instead of your CPU. In this sense, OpenGL refers to your groups of shaders as "programs".
	// Using glCreateProgram creates a shader program and returns a GLuint reference to it.
	program = glCreateProgram();
	glAttachShader(program, vertex_shader);		// This attaches our vertex shader to our program.
	glAttachShader(program, fragment_shader);	// This attaches our fragment shader to our program.

	// This links the program, using the vertex and fragment shaders to create executables to run on the GPU.
	glLinkProgram(program);
	// End of shader and program creation

	// This gets us a reference to the uniform variable in the vertex shader, which is called "MVP".
	// We're using this variable as a 4x4 transformation matrix
	// Only 2 parameters required: A reference to the shader program and the name of the uniform variable within the shader code.
	uniMVP = glGetUniformLocation(program, "MVP");

	// Creates the view matrix using glm::lookAt.
	// First parameter is camera position, second parameter is point to be centered on-screen, and the third paramter is the up axis.
	view = glm::lookAt(	glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	// Creates a projection matrix using glm::perspective.
	// First parameter is the vertical FoV (Field of View), second paramter is the aspect ratio, 3rd parameter is the near clipping plane, 4th parameter is the far clipping plane.
	proj = glm::perspective(45.0f, 800.0f / 600.0f, 0.1f, 100.0f);

	// Allows us to make one less calculation per frame, as long as we don't update the projection and view matrices every frame.
	PV = proj * view;

	// Create your MVP matrices based on the objects' transforms.
	MVP = PV * *obj1->GetTransform();
	MVP2 = PV * *obj2->GetTransform();

	// This is not necessary, but I prefer to handle my vertices in the clockwise order. glFrontFace defines which face of the triangles you're drawing is the front.
	// Essentially, if you draw your vertices in counter-clockwise order, by default (in OpenGL) the front face will be facing you/the screen. If you draw them clockwise, the front face 
	// will face away from you. By passing in GL_CW to this function, we are saying the opposite, and now the front face will face you if you draw in the clockwise order.
	// If you don't use this, just reverse the order of the vertices in your array when you define them so that you draw the points in a counter-clockwise order.
	glFrontFace(GL_CW);

	// This is also not necessary, but more efficient and is generally good practice. By default, OpenGL will render both sides of a triangle that you draw. By enabling GL_CULL_FACE, 
	// we are telling OpenGL to only render the front face. This means that if you rotated the triangle over the X-axis, you wouldn't see the other side of the triangle as it rotated.
	glEnable(GL_CULL_FACE);

	// Determines the interpretation of polygons for rasterization. The first parameter, face, determines which polygons the mode applies to.
	// The face can be either GL_FRONT, GL_BACK, or GL_FRONT_AND_BACK
	// The mode determines how the polygons will be rasterized. GL_POINT will draw points at each vertex, GL_LINE will draw lines between the vertices, and 
	// GL_FILL will fill the area inside those lines.
	glPolygonMode(GL_FRONT, GL_FILL);
}

int main(int argc, char **argv)
{
	// Initializes the GLFW library
	glfwInit();

	// Creates a window given (width, height, title, monitorPtr, windowPtr).
	// Don't worry about the last two, as they have to do with controlling which monitor to display on and having a reference to other windows. Leaving them as nullptr is fine.
	window = glfwCreateWindow(800, 600, "SAT 3D OBB Collision", nullptr, nullptr);

	// Makes the OpenGL context current for the created window.
	glfwMakeContextCurrent(window);
	
	// Sets the number of screen updates to wait before swapping the buffers.
	// Setting this to zero will disable VSync, which allows us to actually get a read on our FPS. Otherwise we'd be consistently getting 60FPS or lower, 
	// since it would match our FPS to the screen refresh rate.
	// Set to 1 to enable VSync.
	glfwSwapInterval(0);

	// Initializes most things needed before the main loop
	init();

	// Enter the main loop.
	while (!glfwWindowShouldClose(window))
	{
		// Call to checkTime() which will determine how to go about updating via a set physics timestep as well as calculating FPS.
		checkTime();

		// Call the render function.
		renderScene();

		// Swaps the back buffer to the front buffer
		// Remember, you're rendering to the back buffer, then once rendering is complete, you're moving the back buffer to the front so it can be displayed.
		glfwSwapBuffers(window);

		// Add one to our frame counter, since we've successfully 
		frame++;

		// Checks to see if any events are pending and then processes them.
		glfwPollEvents();
	}

	// After the program is over, cleanup your data!
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	glDeleteProgram(program);
	// Note: If at any point you stop using a "program" or shaders, you should free the data up then and there.

	delete(obj1);
	delete(obj2);
	delete(cube);

	// Frees up GLFW memory
	glfwTerminate();

	return 0;
}