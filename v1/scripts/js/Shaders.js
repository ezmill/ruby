var RefractionShader = function(){

    this.uniforms = THREE.UniformsUtils.merge([
    { 
        "envMap": { type: "t", value: null },
        "map": { type: "t", value: null },
        "alpha": { type: "t", value: null },
        "flipEnvMap": { type: "f", value: 1.0 },
        "time": { type: "f", value: 0.0 },
        "noiseScale":{type: "f", value:0.0},
        "noiseDetail":{type: "f", value:0.0},
        "refractionRatio":{type: "f", value:0.75},
        "diffuse":{type: "v3", value:new THREE.Vector3(1.0,1.0,1.0)},
        "reflectivity":{type: "f", value:1.0},
        "mouse":{type: "v2", value:null}
      }
  ]);

    this.vertexShader = [

            "varying vec3 vReflect;",
            "uniform float refractionRatio;",
            "uniform sampler2D map;",

            "varying vec2 vUv;",
            "uniform vec2 mouse;",

            "uniform float time;",

            "vec3 transformDirection( in vec3 normal, in mat4 matrix ) {",
            "   return normalize( ( matrix * vec4( normal, 0.0 ) ).xyz );",
            "}",
            "float luminance(vec3 c){",
            "    return dot(c, vec3(.2126, .7152, .0722));",
            "}",
            "void main(){",

            "   vUv = uv;",
            "   vec4 color = texture2D(map, vUv);",
            // "   float depth = smoothstep(0.499,0.501,( color.r + color.g + color.b ) / 3.0);",
            "   float depth = ( color.r + color.g + color.b ) / 3.0;",
            "   float z = ( depth*2.0 - 1.0 ) * (4500.0 - 800.0) + 800.0;",

            "   vec3 pos = vec3(position.x, position.y, z*0.1);",
            // "   vec3 pos = vec3(position.x, position.y, sin(time+uv.x)*z*0.1);",
            // "   vec3 pos = vec3(position.x, position.y, position.z + sin(time*10.0 + position.x*0.1)*100.0);",
            // "   vec3 pos = vec3(position.x, position.y, z*sin(time*10.0 + z)*0.1);",
            "   vec4 mvPosition = modelViewMatrix * vec4( pos, 1.0 );",
            "   gl_Position = projectionMatrix * mvPosition;",
            "       vec3 objectNormal = normal;",
            "   vec4 worldPosition = modelMatrix * vec4( pos, 1.0 );    ",
            "   vec3 worldNormal = transformDirection( objectNormal, modelMatrix );",
            "   vec3 cameraToVertex = normalize( worldPosition.xyz - cameraPosition );",
            "    vReflect = refract( cameraToVertex, worldNormal, refractionRatio );",
            // "    vReflect = refract( cameraToVertex, worldNormal, (mouse.x*0.5 + 0.5)*2.0 );",
            // "   vReflect = reflect( cameraToVertex, worldNormal );",

            "}"

    ].join("\n");

    this.fragmentShader = [

            "uniform float reflectivity;",
            "uniform samplerCube envMap;",
            "uniform sampler2D alpha;",
            "uniform float flipEnvMap;",
            "uniform float mixAmt;",

            "varying vec3 vReflect;",
            "varying vec2 vUv;",
            "uniform vec3 diffuse;",

            "float luminance(vec3 c){",
            "    return dot(c, vec3(.2126, .7152, .0722));",
            "}",
            "void main() {",
            "   vec3 outgoingLight = vec3( 0.0 );",
            "   vec4 diffuseColor = vec4( diffuse, 1.0 );",
            "   float flipNormal = ( -1.0 + 2.0 * float( gl_FrontFacing ) );",
            "   float specularStrength;",
            "   specularStrength = 1.0;",
            "   outgoingLight = diffuseColor.rgb;",

            "   vec3 reflectVec = vReflect;",

            "   vec4 envColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );",
            // "    vec4 envColor = texture2D( map, vUv );",
            // "    envColor.xyz = inputToLinear( envColor.xyz );",
            "   outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );",

            "   gl_FragColor = vec4( outgoingLight, diffuseColor.a );",
            // "   gl_FragColor = vec4( 1.0, 0.0,0.0,1.0 );",
            // "   vec4 alphaTex = texture2D(alpha, vUv);",
            // "   vec4 s = texture2D(map2, vUv);",
            // "   vec4 t = vec4( outgoingLight, diffuseColor.a );",
            // "   vec4 col = mix(t, s, dot(alpha.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(s, alpha,  dot(s.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(alpha, s,  dot(luminance(s.rgb), 1.0/3.0));",
            // "   vec4 col = mix(t, vec4(1.0,0.0,0.0,1.0), dot(alphaTex.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(s, t, mixAmt);",
            // "    gl_FragColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );",

            "}"

        ].join("\n")
}
var AShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "BUF_A"  : { type: "t", value: null },
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "FRAME"  : { type: "f", value: null },
                "resolution"  : { type: "v2", value: null },
                "mouse"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D BUF_A;",
            "uniform sampler2D BUF_B;",
            "uniform sampler2D alpha;",
            "uniform float FRAME;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "void main()",
            "{",
            "    const float _K0 = -20.0/6.0; // center weight",
            "    const float _K1 = 4.0/6.0; // edge-neighbors",
            "    const float _K2 = 1.0/6.0; // vertex-neighbors",
            "    const float cs = -0.5; // curl scale",
            "    const float ls = 0.03; // laplacian scale",
            "    const float ps = -0.05; // laplacian of divergence scale",
            "    const float ds = 0.05; // divergence scale",
            "    const float is = 0.01; // image derivative scale",
            "    const float pwr = 1.0; // power when deriving rotation angle from curl",
            "    const float amp = 1.0; // self-amplification",
            "    const float sq2 = 0.7; // diagonal weight",

            "    vec2 texel = mouse.y*10.0 / resolution.xy;",
            "    ",
            "    // 3x3 neighborhood coordinates",
            "    float step_x = texel.x;",
            "    float step_y = texel.y;",
            "    vec2 n  = vec2(0.0, step_y);",
            "    vec2 ne = vec2(step_x, step_y);",
            "    vec2 e  = vec2(step_x, 0.0);",
            "    vec2 se = vec2(step_x, -step_y);",
            "    vec2 s  = vec2(0.0, -step_y);",
            "    vec2 sw = vec2(-step_x, -step_y);",
            "    vec2 w  = vec2(-step_x, 0.0);",
            "    vec2 nw = vec2(-step_x, step_y);",
            "    ",
            "    // sobel filter",
            "    vec3 im = texture2D(BUF_B, vUv).xyz;",
            "    vec3 im_n = texture2D(BUF_B, vUv+n).xyz;",
            "    vec3 im_e = texture2D(BUF_B, vUv+e).xyz;",
            "    vec3 im_s = texture2D(BUF_B, vUv+s).xyz;",
            "    vec3 im_w = texture2D(BUF_B, vUv+w).xyz;",
            "    vec3 im_nw = texture2D(BUF_B, vUv+nw).xyz;",
            "    vec3 im_sw = texture2D(BUF_B, vUv+sw).xyz;",
            "    vec3 im_ne = texture2D(BUF_B, vUv+ne).xyz;",
            "    vec3 im_se = texture2D(BUF_B, vUv+se).xyz;",

            "    float dx = 3.0 * (length(im_e) - length(im_w)) + (length(im_ne) + length(im_se) - length(im_sw) - length(im_nw));",
            "    float dy = 3.0 * (length(im_n) - length(im_s)) + (length(im_nw) + length(im_ne) - length(im_se) - length(im_sw));",

            "    // vector field neighbors",
            "    vec3 uv =    texture2D(BUF_A, vUv).xyz;",
            "    vec3 uv_n =  texture2D(BUF_A, vUv+n).xyz;",
            "    vec3 uv_e =  texture2D(BUF_A, vUv+e).xyz;",
            "    vec3 uv_s =  texture2D(BUF_A, vUv+s).xyz;",
            "    vec3 uv_w =  texture2D(BUF_A, vUv+w).xyz;",
            "    vec3 uv_nw = texture2D(BUF_A, vUv+nw).xyz;",
            "    vec3 uv_sw = texture2D(BUF_A, vUv+sw).xyz;",
            "    vec3 uv_ne = texture2D(BUF_A, vUv+ne).xyz;",
            "    vec3 uv_se = texture2D(BUF_A, vUv+se).xyz;",
            "    ",
            "    // uv.x and uv.y are our x and y components, uv.z is divergence ",

            "    // laplacian of all components",
            "    vec3 lapl  = _K0*uv + _K1*(uv_n + uv_e + uv_w + uv_s) + _K2*(uv_nw + uv_sw + uv_ne + uv_se);",
            "    float sp = ps * lapl.z;",
            "    ",
            "    // calculate curl",
            "    // vectors point clockwise about the center point",
            "    float curl = uv_n.x - uv_s.x - uv_e.y + uv_w.y + sq2 * (uv_nw.x + uv_nw.y + uv_ne.x - uv_ne.y + uv_sw.y - uv_sw.x - uv_se.y - uv_se.x);",
            "    ",
            "    // compute angle of rotation from curl",
            "    float sc = cs * sign(curl) * pow(abs(curl), pwr);",
            "    ",
            "    // calculate divergence",
            "    // vectors point inwards towards the center point",
            "    float div  = uv_s.y - uv_n.y - uv_e.x + uv_w.x + sq2 * (uv_nw.x - uv_nw.y - uv_ne.x - uv_ne.y + uv_sw.x + uv_sw.y + uv_se.y - uv_se.x);",
            "    float sd = ds * div;",

            "    vec2 norm = normalize(uv.xy);",
            "    ",
            "    // temp values for the update rule",
            "    float ta = amp * uv.x + ls * lapl.x + norm.x * sp + uv.x * sd + is * dx;",
            "    float tb = amp * uv.y + ls * lapl.y + norm.y * sp + uv.y * sd + is * dy;",

            "    // rotate",
            "    float a = ta * cos(sc) - tb * sin(sc);",
            "    float b = ta * sin(sc) + tb * cos(sc);",
            "    ",
            "    // initialize with noise",
            // "    if(FRAME<10.0) {",
            // "        gl_FragColor = -0.5 + texture2D(BUF_B, vUv);",
            // "    } else {",
            // "        gl_FragColor = clamp(vec4(a,b,div,1), -1., 1.);",
            // "    }",
            "    vec4 col = clamp(vec4(a,b,div,1), -1., 1.);",
            "    vec4 inputTex = -0.5 + texture2D(BUF_B, vUv);",
            "    vec4 alpha = texture2D(alpha, vUv);",
            "    // initialize with image",
            "    if(FRAME<10.0) {",
            "        gl_FragColor = inputTex;",
            "    } else {",
            // "        gl_FragColor = vec4(clamp(im + ds * diffusion_im, 0.0, 1.0), 0.0);",
            "        gl_FragColor = col;",
            // "        gl_FragColor = mix(inputTex, col, dot(alpha.rgb, vec3(1.0))/3.0);",
            "    }",
            "    ",

            "}",
        
        ].join("\n");
}
var BShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "INPUT"  : { type: "t", value: null },
                "BUF_A"  : { type: "t", value: null },
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "FRAME"  : { type: "f", value: null },
                "resolution"  : { type: "v2", value: null },
                "mouse"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D INPUT;",
            "uniform sampler2D BUF_A;",
            "uniform sampler2D BUF_B;",
            "uniform sampler2D alpha;",
            "uniform float FRAME;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "void main()",
            "{",
            "    const float ds = 2.0; // diffusion rate",
            "    const float darken = 0.0; // darkening",
            "    const float D1 = 0.2;  // edge neighbors",
            "    const float D2 = 0.05; // vertex neighbors",
            "    ",
            "    vec2 texel = mouse.y*10.0 / resolution.xy;",
            "    ",
            "    // 3x3 neighborhood coordinates",
            "    float step_x = texel.x;",
            "    float step_y = texel.y;",
            "    vec2 n  = vec2(0.0, step_y);",
            "    vec2 ne = vec2(step_x, step_y);",
            "    vec2 e  = vec2(step_x, 0.0);",
            "    vec2 se = vec2(step_x, -step_y);",
            "    vec2 s  = vec2(0.0, -step_y);",
            "    vec2 sw = vec2(-step_x, -step_y);",
            "    vec2 w  = vec2(-step_x, 0.0);",
            "    vec2 nw = vec2(-step_x, step_y);",
            "    ",
            "    vec3 components = texture2D(BUF_A, vUv).xyz;",
            "    ",
            "    float a = components.x;",
            "    float b = components.y;",
            "    ",
            "    vec3 im =    texture2D(BUF_B, vec2(mouse)*0.001+vUv).xyz;",
            "    vec3 im_n =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+n).xyz;",
            "    vec3 im_e =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+e).xyz;",
            "    vec3 im_s =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+s).xyz;",
            "    vec3 im_w =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+w).xyz;",
            "    vec3 im_nw = texture2D(BUF_B, vec2(mouse)*0.001+vUv+nw).xyz;",
            "    vec3 im_sw = texture2D(BUF_B, vec2(mouse)*0.001+vUv+sw).xyz;",
            "    vec3 im_ne = texture2D(BUF_B, vec2(mouse)*0.001+vUv+ne).xyz;",
            "    vec3 im_se = texture2D(BUF_B, vec2(mouse)*0.001+vUv+se).xyz;",

            "    float D1_e = D1 * a;",
            "    float D1_w = D1 * -a;",
            "    float D1_n = D1 * b;",
            "    float D1_s = D1 * -b;",
            "    float D2_ne = D2 * (b + a);",
            "    float D2_nw = D2 * (b - a);",
            "    float D2_se = D2 * (a - b);",
            "    float D2_sw = D2 * (- a - b);",

            "    vec3 diffusion_im = -darken * length(vec2(a, b)) * im + im_n*D1_n + im_ne*D2_ne + im_e*D1_e + im_se*D2_se + im_s*D1_s + im_sw*D2_sw + im_w*D1_w + im_nw*D2_nw;",

            "    vec4 col = vec4(clamp(im + (mouse.x*1.0) * diffusion_im, 0.0, 1.0), 0.0);",
            "    vec4 inputTex = texture2D(INPUT, vUv);",
            "    vec4 alpha = texture2D(alpha, vUv);",
            "    // initialize with image",
            "    if(FRAME<10.0) {",
            "        gl_FragColor = inputTex;",
            "    } else {",
            // "        gl_FragColor = vec4(clamp(im + ds * diffusion_im, 0.0, 1.0), 0.0);",
            // "        gl_FragColor = mix(inputTex, col, dot(alpha.rgb, vec3(1.0))/3.0);",
            "        gl_FragColor = col;",
            "    }",
            "}",
        
        ].join("\n");
}
var OUTPUTShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D BUF_B;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            
            "void main()",
            "{",
            "   vec3 col = texture2D(BUF_B, vUv).rgb;",
            "   gl_FragColor = vec4(col*vec3(1.0),1.0);",
            "}",
 
        ].join("\n");
}
var PassShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D texture;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            
            "void main()",
            "{",
            "   vec3 col = texture2D(texture, vUv).rgb;",
            "   gl_FragColor = vec4(col,1.0);",
            "}",
 
        ].join("\n");
}
var DifferencingShader = function(){
    this.uniforms = THREE.UniformsUtils.merge( [

        {
            "texture"  : { type: "t", value: null },
            "mouse"  : { type: "v2", value: null },
            "resolution"  : { type: "v2", value: null },
            "time"  : { type: "f", value: null },
            "texture2"  : { type: "t", value: null },
        }
    ] ),

    this.vertexShader = [

        "varying vec2 vUv;",
        "void main() {",
        "    vUv = uv;",
        "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
        "}"
    
    ].join("\n"),
    
    this.fragmentShader = [
        
        "uniform sampler2D texture;",
        "uniform sampler2D texture2;",
        "uniform vec2 resolution;",
        "uniform vec2 mouse;",
        "uniform float time;",
        "varying vec2 vUv;",

        "void main() {",
        "  vec4 tex0 = texture2D(texture, vUv);",
        "  vec4 tex1 = texture2D(texture2, vUv);",
        "  vec4 fc = (tex1 - tex0);",
        "  gl_FragColor = vec4(fc);",
        "}"
    
    ].join("\n")
    
}
var WarpShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);

        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D texture;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",

            "float rand(vec2 p)",
            "{",
            "    vec2 n = floor(p/2.0);",
            "     return fract(cos(dot(n,vec2(48.233,39.645)))*375.42); ",
            "}",
            "float srand(vec2 p)",
            "{",
            "     vec2 f = floor(p);",
            "    vec2 s = smoothstep(vec2(0.0),vec2(1.0),fract(p));",
            "    ",
            "    return mix(mix(rand(f),rand(f+vec2(1.0,0.0)),s.x),",
            "           mix(rand(f+vec2(0.0,1.0)),rand(f+vec2(1.0,1.0)),s.x),s.y);",
            "}",
            "float noise(vec2 p)",
            "{",
            "     float total = srand(p/128.0)*0.5+srand(p/64.0)*0.35+srand(p/32.0)*0.1+srand(p/16.0)*0.05;",
            "    return total;",
            "}",

            "void main()",
            "{",
            "    float t = time;",
            "    vec2 warp = vec2(noise(gl_FragCoord.xy+t)+noise(gl_FragCoord.xy*0.5+t*3.5),",
            "                     noise(gl_FragCoord.xy+128.0-t)+noise(gl_FragCoord.xy*0.6-t*2.5))*0.5-0.25;",
            // "    vec2 uv = gl_FragCoord.xy / resolution.xy+warp;",
            "    vec2 mW = warp*vec2(-1.0 * mouse);",
            "    vec2 uv = vUv+mW*sin(time);",
            "    vec4 look = texture2D(texture,uv);",
            "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*vec2(mouse.x*uv.x/10.0, mouse.y*uv.y/10.0);",
            "    vec2 coord = offs+vUv;",
            "    vec4 repos = texture2D(texture, uv);",
                "gl_FragColor = vec4(repos.rgb,1.0);",
            "}"


        
        ].join("\n");
}