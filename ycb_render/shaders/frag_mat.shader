#version 460
in vec3 Normal;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_obj;
in float inverse_normal;
uniform vec3 mat_ambient;
uniform vec3 mat_diffuse;
uniform vec3 mat_specular;
uniform float mat_shininess;

layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCObject;
layout (location = 4) out vec4 PCColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color
void main() {
    if (inverse_normal > 0) discard; // discard the wrong pixel
    vec3 norm = normalize(Normal);
    vec3 ambient =  mat_ambient * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light_color * mat_diffuse;
    vec3 viewDir = normalize(Pos_cam - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), mat_shininess);
    vec3 specular = light_color * (spec * mat_specular);     
    outputColour =  vec4(ambient + diffuse + specular, 1);

    //NormalColour =  vec4((Normal_cam + 1) / 2,1);
    NormalColour = vec4(Normal_cam,1);
    InstanceColour = vec4(Instance_color,1);
    PCObject = vec4(Pos_obj,1);
    PCColour = vec4(Pos_cam,1);
}