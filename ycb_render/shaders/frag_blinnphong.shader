#version 460
uniform sampler2D texUnit;
in vec2 theCoords;
in vec3 Normal;
in vec3 Normal_cam;
in vec3 FragPos;
in vec3 Instance_color;
in vec3 Pos_cam;
in vec3 Pos_obj;
in float inverse_normal;
layout (location = 0) out vec4 outputColour;
layout (location = 1) out vec4 NormalColour;
layout (location = 2) out vec4 InstanceColour;
layout (location = 3) out vec4 PCObject;
layout (location = 4) out vec4 PCColour;

uniform vec3 light_position;  // in world coordinate
uniform vec3 light_color; // light color
uniform vec3 world_light_pos1;
uniform vec3 world_light_pos2;
uniform vec3 mat_ambient;
uniform vec3 mat_diffuse;
uniform vec3 mat_specular;
uniform float mat_shininess;

void main() {
    if (inverse_normal > 0) discard;
    vec4 texColor = texture(texUnit, theCoords); 
    if(texColor.a < 0.1) discard; 

    // attenuation
    float a = 1.0f;
    float b =  0.5f;
    float c =  0.25f;
    float r = length(light_position - Pos_obj);
    float scalar = (a + b*r + c*r*r);
    if(scalar < 0.00000001)
        scalar = 0.0; 
    else
        scalar = 1.0/scalar;

    vec3 norm = normalize(Normal);
    vec3 ambient =  mat_ambient * light_color;
    vec3 lightDir = normalize(light_position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light_color * mat_diffuse;
    vec3 viewDir = normalize(Pos_cam - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), mat_shininess);
    vec3 specular = light_color * (spec * mat_specular);   
         
    // gamma correction
    vec3 linearColour =  ambient + scalar*(diffuse +  specular);
    vec3 gamma = vec3(1.0/2.2);
    outputColour =  texColor * vec4(pow(linearColour, gamma), 1);
    
    // add few more lights
    // lightDir = normalize(world_light_pos1 - FragPos);
    // diff = max(dot(norm, lightDir), 0.0);
    // diffuse = diff * light_color * mat_diffuse;
    // viewDir = normalize(Pos_cam - FragPos);
    // reflectDir = reflect(-lightDir, norm);  
    // spec = pow(max(dot(viewDir, reflectDir), 0.0), mat_shininess);
    // specular = light_color * (spec * mat_specular);              
    // outputColour +=  texture(texUnit, theCoords)  * vec4(diffuse + ambient + specular, 1);
    // lightDir = normalize(world_light_pos2 - FragPos);
    // diff = max(dot(norm, lightDir), 0.0);
    // diffuse = diff * light_color * mat_diffuse;
    // viewDir = normalize(Pos_cam - FragPos);
    // reflectDir = reflect(-lightDir, norm);  
    // spec = pow(max(dot(viewDir, reflectDir), 0.0), mat_shininess);
    // specular = light_color * (spec * mat_specular);              
    // outputColour +=  texture(texUnit, theCoords)  * vec4(ambient + scalar*(diffuse +  specular), 1);

    //NormalColour =  vec4((Normal_cam + 1) / 2,1);
    NormalColour = vec4(Normal_cam,1);
    InstanceColour = vec4(Instance_color,1);
    PCObject = vec4(Pos_obj,1);
    PCColour = vec4(Pos_cam,1);
}