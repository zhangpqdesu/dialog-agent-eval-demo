'use strict';
// WebGL Dancing Azure Sky — 改编自 Shadertoy "Dancing Azure Sky"
(function () {
  const canvas = document.getElementById('shaderCanvas');
  if (!canvas) return;

  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  if (!gl) return;

  // ── Vertex Shader ──
  const VS = `
    attribute vec2 aPos;
    void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
  `;

  // ── Fragment Shader（改编自 Shadertoy，iTime/iResolution → uTime/uRes）──
  const FS = `
    precision mediump float;
    uniform float uTime;
    uniform vec2  uRes;

    float random(in vec2 st) {
      return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    float noise(in vec2 st) {
      vec2 i = floor(st);
      vec2 f = fract(st);
      float a = random(i);
      float b = random(i + vec2(1.0, 0.0));
      float c = random(i + vec2(0.0, 1.0));
      float d = random(i + vec2(1.0, 1.0));
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(a, b, u.x)
           + (c - a) * u.y * (1.0 - u.x)
           + (d - b) * u.x * u.y;
    }

    #define NUM_OCTAVES 5
    float fbm(in vec2 st) {
      float v = 0.0;
      float a = 0.5;
      vec2 shift = vec2(100.0);
      mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
      for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(st);
        st = rot * st * 2.0 + shift;
        a *= 0.5;
      }
      return v;
    }

    void main() {
      vec2 fragCoord = gl_FragCoord.xy;
      vec2 st = fragCoord / uRes.xy;
      st.x *= uRes.x / uRes.y;

      float t = uTime * 0.3;

      vec2 q;
      q.x = fbm(st + 0.00 * t);
      q.y = fbm(st + vec2(1.0));

      vec2 r;
      r.x = fbm(st + 1.0 * q + vec2(1.7, 9.2) + 0.15 * t);
      r.y = fbm(st + 1.0 * q + vec2(8.3, 2.8) + 0.126 * t);

      float f = fbm(st + r);

      vec3 colorBase   = vec3(0.02, 0.05, 0.2);
      vec3 colorMid    = vec3(0.10, 0.40, 0.8);
      vec3 colorHigh   = vec3(0.00, 0.85, 0.9);
      vec3 colorAccent = vec3(0.5,  0.1,  0.7);

      vec3 col = mix(colorBase,   colorMid,    clamp(f * f * 3.0, 0.0, 1.0));
      col      = mix(col,         colorAccent, clamp(length(q) * 0.5, 0.0, 1.0));
      col      = mix(col,         colorHigh,   clamp(length(r.x) * 1.2 - 0.2, 0.0, 1.0));
      col     *= (f * f * f + 0.8 * f * f + 0.6 * f);
      col     *= 1.5;

      float stars = pow(random(st * 200.0 + uTime * 0.01), 150.0);
      col += vec3(stars) * clamp(1.0 - f * 1.5, 0.0, 1.0);

      vec2 center = vec2(0.5 * (uRes.x / uRes.y), 0.5);
      float vig = 1.0 - smoothstep(0.4, 1.5, length(st - center));
      col *= vig;

      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }
  `;

  function compile(src, type) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    return sh;
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(VS, gl.VERTEX_SHADER));
  gl.attachShader(prog, compile(FS, gl.FRAGMENT_SHADER));
  gl.linkProgram(prog);
  gl.useProgram(prog);

  // 全屏四边形
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  const aPos = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const uTime = gl.getUniformLocation(prog, 'uTime');
  const uRes  = gl.getUniformLocation(prog, 'uRes');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  window.addEventListener('resize', resize);
  resize();

  let start = null;
  function frame(ts) {
    if (!start) start = ts;
    const t = (ts - start) / 1000;
    gl.uniform1f(uTime, t);
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
