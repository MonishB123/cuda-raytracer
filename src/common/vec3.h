#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

class vec3 {
public:
    HOST_DEVICE vec3() : e{0,0,0} {}
    HOST_DEVICE vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    HOST_DEVICE float x() const { return e[0]; }
    HOST_DEVICE float y() const { return e[1]; }
    HOST_DEVICE float z() const { return e[2]; }

    HOST_DEVICE vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    HOST_DEVICE float operator[](int i) const { return e[i]; }
    HOST_DEVICE float& operator[](int i) { return e[i]; }

    HOST_DEVICE vec3& operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    HOST_DEVICE vec3& operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    HOST_DEVICE vec3& operator/=(const float t) {
        return *this *= 1/t;
    }

    HOST_DEVICE float length() const {
        return sqrtf(length_squared());
    }

    HOST_DEVICE float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    float e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

HOST_DEVICE inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HOST_DEVICE inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HOST_DEVICE inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HOST_DEVICE inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

HOST_DEVICE inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

HOST_DEVICE inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

HOST_DEVICE inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

HOST_DEVICE inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HOST_DEVICE inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

HOST_DEVICE inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

#endif
