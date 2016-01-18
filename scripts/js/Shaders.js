var VectorFieldShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "noise"  : { type: "t", value: null },
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
            "uniform sampler2D noise;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "float hash( vec2 p )",
            "{",
            "    float h = dot(p,vec2(127.1,311.7));",
            "    ",
            "    return -1.0 + 2.0*fract(sin(h)*43758.5453123);",
            "}",

            "float noisef( in vec2 p )",
            "{",
            "    vec2 i = floor( p );",
            "    vec2 f = fract( p );",
            "    ",
            "    vec2 u = f*f*(3.0-2.0*f);",

            "    return mix( mix( hash( i + vec2(0.0,0.0) ), ",
            "                     hash( i + vec2(1.0,0.0) ), u.x),",
            "                mix( hash( i + vec2(0.0,1.0) ), ",
            "                     hash( i + vec2(1.0,1.0) ), u.x), u.y);",
            "}",
            "#define SIZE (resolution.x/128.0) // cell size in texture coordinates",
            // "#define ZOOM (40. *256./resolution.x)",
            // "#define ZOOM 1.0",
            "#define ZOOM 1.0 + (mouse.x*0.5 + 1.0)*10.0",
            "float STRIP  = 1.;    // nbr of parallel lines per cell",
            "float V_ADV  = 0.1;    // velocity",
            "float V_BOIL = 0.5;    // change speed",

            "bool CONT , FLOW ,ATTRAC; // A: draw field or attractor ?",
            "vec3 flow( vec2 uv) {",
            // "   "
            "   vec2 iuv = vec2(noisef(floor(SIZE*(uv)+.5)/SIZE));",
            "	vec2 fuv = 2.*SIZE*(uv-iuv);",
            "	vec2 pos = .01*V_ADV*vec2(cos(time)+sin(.356*time)+2.*cos(.124*time),sin(.854*time)+cos(.441*time)+2.*cos(.174*time));	if (CONT) iuv=uv;",
            "	vec3 tex = 2.*texture2D(noise,iuv/(ZOOM*SIZE)-pos).rgb-1.;",
            "	float ft = fract(time*V_BOIL)*3.;",
            "	if      (ft<1.) tex = mix(tex.rgb,tex.gbr,ft);",
            "	else if (ft<2.) tex = mix(tex.gbr,tex.brg,ft-1.);",
            "	else            tex = mix(tex.brg,tex.rgb,ft-2.);",
            "	return (FLOW) ? vec3(tex.y,-tex.x,tex.z): tex;",
            "}",

            "void main()",
            "{",
            " 	CONT   = true; // C: is field interpolated in cells ?",
            " 	FLOW   = false; // F: flow or gradient ?",
            " 	ATTRAC = true; // A: draw field or attractor ?",

            "   vec2 uv = vUv;",
            "	vec3 col;",
            "   vec2 uv0 = vUv;",
            "    float f = 0.0;",
            "    uv0 *= 5.0;",
            "       mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );",
            "    f  = 0.5000*noisef( uv0 ); uv0 = m*uv0;",
            "    f += 0.2500*noisef( uv0 ); uv0 = m*uv0;",
            "    f += 0.1250*noisef( uv0 ); uv0 = m*uv0;",
            "    f += 0.0625*noisef( uv0 ); uv0 = m*uv0;",
            "    f = 0.5 + 0.5*f;",
            "    if (ATTRAC) {",
            "    	vec2 tex = uv;",
            "    	for(int i=0; i<15; i++) ",
            "           tex += (f*0.003*mouse.xy)*flow(tex).xy;",
            "    		col = texture2D(texture,tex).rgb;",
            "    } else {    ",
            "   		vec2 iuv = floor(SIZE*(uv)+.5)/SIZE;",
            "		vec2 fuv = 2.*SIZE*(uv-iuv);",
            "    	vec3 tex = flow(uv);",
            "   		float v = fuv.x*tex.x+fuv.y*tex.y;     ",
            "		// v = length(fuv);",
            "		v = sin(STRIP*v);",
            "		col = vec3(1.-v*v*SIZE) * mix(tex,vec3(1.),.5);",
            "    }",

            "	gl_FragColor = vec4(col,1.0);",
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
        // "  gl_FragColor = vec4(fc);",
        "  gl_FragColor = vec4(tex1);",
        "}"
    
    ].join("\n")
    
}

var GradientShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "texture2"  : { type: "t", value: null },
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
            "uniform sampler2D texture2;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "float hash( vec2 p )",
            "{",
            "    float h = dot(p,vec2(127.1,311.7));",
            "    ",
            "    return -1.0 + 2.0*fract(sin(h)*43758.5453123);",
            "}",

            "float noise( in vec2 p )",
            "{",
            "    vec2 i = floor( p );",
            "    vec2 f = fract( p );",
            "    ",
            "    vec2 u = f*f*(3.0-2.0*f);",

            "    return mix( mix( hash( i + vec2(0.0,0.0) ), ",
            "                     hash( i + vec2(1.0,0.0) ), u.x),",
            "                mix( hash( i + vec2(0.0,1.0) ), ",
            "                     hash( i + vec2(1.0,1.0) ), u.x), u.y);",
            "}",
            "void main()",
            "{",
            "   vec2 uv = vUv;",
            "   vec2 uv0 = vUv;",
            "    float f = 0.0;",
            "    uv0 *= 5.0;",
            "       mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );",
            "    f  = 0.5000*noise( uv0 ); uv0 = m*uv0;",
            "    f += 0.2500*noise( uv0 ); uv0 = m*uv0;",
            "    f += 0.1250*noise( uv0 ); uv0 = m*uv0;",
            "    f += 0.0625*noise( uv0 ); uv0 = m*uv0;",
            "    f = 0.5 + 0.5*f;",
            // "   vec3 col = vec3(uv,0.5+0.5*sin(time));",
            // "vec3 col = mix(vec3(0.5+0.5*cos(time), 1.0 - uv.yx), vec3(uv.yx,0.5+0.5*sin(time)), (0.5 + 0.5*sin((-1.0 + uv.x*2.0)*(-1.0 + uv.y*2.0)*20.0)));",
            // "vec3 col = mix(vec3(0.5+0.5*cos(time), 1.0 - uv.yx), vec3(uv.yx,0.5+0.5*sin(time)), f);",
            // "vec3 col = mix(texture2D(texture, vUv).rgb, texture2D(texture2, vUv).rgb, (0.5 + 0.5*sin((-1.0 + uv.x*2.0)*(-1.0 + uv.y*2.0)*20.0)));",
            "vec3 col = mix(texture2D(texture, vUv).rgb, texture2D(texture2, vUv).rgb, f);",
            "   if(mod(gl_FragCoord.x, 0.003) == 0.0){",
            "       if(mod(gl_FragCoord.y, 0.003) == 0.0){",
            // "           col = vec3(1.);",
            "       }",
            "   }",
            "   gl_FragColor = vec4(col,1.0);",
            "}",
 
        ].join("\n");
}
var ColorShader = function(){
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

            "vec3 rainbow(float h) {",
            "  h = mod(mod(h, 1.0) + 1.0, 1.0);",
            "  float h6 = h * 6.0;",
            "  float r = clamp(h6 - 4.0, 0.0, 1.0) +",
            "    clamp(2.0 - h6, 0.0, 1.0);",
            "  float g = h6 < 2.0",
            "    ? clamp(h6, 0.0, 1.0)",
            "    : clamp(4.0 - h6, 0.0, 1.0);",
            "  float b = h6 < 4.0",
            "    ? clamp(h6 - 2.0, 0.0, 1.0)",
            "    : clamp(6.0 - h6, 0.0, 1.0);",
            "  return vec3(r, g, b);",
            "}",

            "vec3 rgb2hsv(vec3 c)",
            "{",
            "    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);",
            "    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));",
            "    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));",
            "    ",
            "    float d = q.x - min(q.w, q.y);",
            "    float e = 1.0e-10;",
            "    return vec3(abs(( (q.z + (q.w - q.y) / (6.0 * d + e))) ), d / (q.x + e), q.x);",
            "}",

            "vec3 hsv2rgb(vec3 c)",
            "{",
            "    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);",
            "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);",
            "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);",
            "}",
            "void main()",
            "{",
            "  vec4 tex0 = texture2D(texture, vUv);",
            "  vec3 hsv = rgb2hsv(tex0.rgb);",
            "  hsv.r += 0.01;",
            "  hsv.g *= 1.001;",
            "  vec3 rgb = hsv2rgb(hsv); ",
            "   gl_FragColor = vec4(rgb,1.0);",
            "}"
        ].join("\n");
}
var RgbShiftShader = function(){
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

            "void main() {",

                "float ChromaticAberration = 10.0 / 10.0 + 8.0;",
                "vec2 uv = vUv;",

                "vec2 texel = 1.0 / resolution.xy;",

                "vec2 coords = (uv - 0.5) * 2.0;",
                "float coordDot = dot (coords, coords);",

                "vec2 precompute = ChromaticAberration * coordDot * coords;",
                "vec2 uvR = uv - texel.xy * precompute;",
                "vec2 uvB = uv + texel.xy * precompute;",

                "vec4 color;",
                "float distance = 0.1;",
                "float speed = 0.5;",
                "vec2 rCoord = vec2(uvR.x + cos(time*speed)*distance, uvR.y + sin(time*speed)*distance);",
                "vec2 bCoord = vec2(uvB.x + sin(time*speed)*distance, uvB.y + cos(time*speed)*distance);",
                "color.r = texture2D(texture, rCoord).r;",
                "color.g = texture2D(texture, uv).g;",
                "color.b = texture2D(texture, bCoord).b;",
                "gl_FragColor = vec4(color.rgb,1.0);",
            "}"


        
        ].join("\n");
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
var CurvesShader = function(red, green, blue){
        function clamp(lo, value, hi) {
            return Math.max(lo, Math.min(value, hi));
        }
        function splineInterpolate(points) {
            var interpolator = new SplineInterpolator(points);
            var array = [];
            for (var i = 0; i < 256; i++) {
                array.push(clamp(0, Math.floor(interpolator.interpolate(i / 255) * 256), 255));
            }
            return array;
        }

        red = splineInterpolate(red);
        if (arguments.length == 1) {
            green = blue = red;
        } else {
            green = splineInterpolate(green);
            blue = splineInterpolate(blue);
        }
        // createCanvas(red, green, blue);
        var array = [];
        for (var i = 0; i < 256; i++) {
            array.splice(array.length, 0, red[i], green[i], blue[i], 255);
        }
        // console.log(array);
        curveMap = new THREE.DataTexture(array, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
        curveMap.minFilter = curveMap.magFilter = THREE.LinearFilter;
        curveMap.needsUpdate = true;
        // var noiseSize = 256;
        var size = 256;
        var data = new Uint8Array( 4 * size );
        for ( var i = 0; i < size * 4; i ++ ) {
            data[ i ] = array[i] | 0;
        }
        var dt = new THREE.DataTexture( data, 256, 1, THREE.RGBAFormat );
        // dt.wrapS = THREE.ClampToEdgeWrapping;
        // dt.wrapT = THREE.ClampToEdgeWrapping;
        dt.needsUpdate = true;
        // console.log(dt);
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "curveMap"  : { type: "t", value: dt },
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
            "uniform sampler2D curveMap;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",

            "void main(){",
            
            "   vec4 curveColor = texture2D(texture, vUv);",
            "   curveColor.r = texture2D(curveMap, vec2(curveColor.r)).r;",
            "   curveColor.g = texture2D(curveMap, vec2(curveColor.g)).g;",
            "   curveColor.b = texture2D(curveMap, vec2(curveColor.b)).b;",

            "   gl_FragColor = vec4(curveColor.rgb,1.0);",
            "}",


        
        ].join("\n");
}

function SplineInterpolator(points) {
    var n = points.length;
    this.xa = [];
    this.ya = [];
    this.u = [];
    this.y2 = [];

    points.sort(function(a, b) {
        return a[0] - b[0];
    });
    for (var i = 0; i < n; i++) {
        this.xa.push(points[i][0]);
        this.ya.push(points[i][1]);
    }

    this.u[0] = 0;
    this.y2[0] = 0;

    for (var i = 1; i < n - 1; ++i) {
        // This is the decomposition loop of the tridiagonal algorithm. 
        // y2 and u are used for temporary storage of the decomposed factors.
        var wx = this.xa[i + 1] - this.xa[i - 1];
        var sig = (this.xa[i] - this.xa[i - 1]) / wx;
        var p = sig * this.y2[i - 1] + 2.0;

        this.y2[i] = (sig - 1.0) / p;

        var ddydx = 
            (this.ya[i + 1] - this.ya[i]) / (this.xa[i + 1] - this.xa[i]) - 
            (this.ya[i] - this.ya[i - 1]) / (this.xa[i] - this.xa[i - 1]);

        this.u[i] = (6.0 * ddydx / wx - sig * this.u[i - 1]) / p;
    }

    this.y2[n - 1] = 0;

    // This is the backsubstitution loop of the tridiagonal algorithm
    for (var i = n - 2; i >= 0; --i) {
        this.y2[i] = this.y2[i] * this.y2[i + 1] + this.u[i];
    }
}

SplineInterpolator.prototype.interpolate = function(x) {
    var n = this.ya.length;
    var klo = 0;
    var khi = n - 1;

    // We will find the right place in the table by means of
    // bisection. This is optimal if sequential calls to this
    // routine are at random values of x. If sequential calls
    // are in order, and closely spaced, one would do better
    // to store previous values of klo and khi.
    while (khi - klo > 1) {
        var k = (khi + klo) >> 1;

        if (this.xa[k] > x) {
            khi = k; 
        } else {
            klo = k;
        }
    }

    var h = this.xa[khi] - this.xa[klo];
    var a = (this.xa[khi] - x) / h;
    var b = (x - this.xa[klo]) / h;

    // Cubic spline polynomial is now evaluated.
    return a * this.ya[klo] + b * this.ya[khi] + 
        ((a * a * a - a) * this.y2[klo] + (b * b * b - b) * this.y2[khi]) * (h * h) / 6.0;
};
var MedianBlurShader = function(){
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

            "void main() {",
                "vec4 center = texture2D(texture, vUv);",
                "float exponent = 1.0;",
                "vec4 color = vec4(0.0);",
                "float total = 0.0;",
                "for (float x = -4.0; x <= 4.0; x += 2.0) {",
                "    for (float y = -4.0; y <= 4.0; y += 2.0) {",
                "        vec4 sample = texture2D(texture, vUv + vec2(x, y) / resolution);",
                "        float weight = 1.0 - abs(dot(sample.rgb - center.rgb, vec3(0.25)));",
                "        weight = pow(weight, exponent);",
                "        color += sample * weight;",
                "        total += weight;",
                "    }",
                "}",
                "vec4 col2 = color / total;",
                "gl_FragColor = vec4(col2.rgb,1.0);",
            "}"


        
        ].join("\n");
}
