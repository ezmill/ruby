function Screensaver(SCENE, CAMERA, RENDERER, TEXTURES){
	this.scene = SCENE;
	this.camera = CAMERA;
	this.renderer = RENDERER;
	this.textures = TEXTURES;
	this.texturePlanes = [];
	this.refractionPlaneGeometry;
	this.refractionPlaneMaterial;
	this.refractionPlane;
	this.init = function(){

		// for(var i = 0; i < this.textures.length; i++){
			// this.texturePlanes[i] = new TexturePlane(this.scene, this.camera, this.renderer, this.textures[i], Math.random());
			// this.texturePlanes[i].init();
		// }

		this.cubeCamera = new THREE.CubeCamera( 1, 100000, 2048 );
		this.cubeCamera.renderTarget.mapping = THREE.CubeRefractionMapping;
		this.scene.add(this.cubeCamera);



		this.refractionPlaneGeometry = new THREE.PlaneGeometry(renderSize.x*1.1, renderSize.y*1.1, 500, 500);
		// this.refractionPlaneGeometry = new THREE.SphereGeometry(500,250,250);
		// this.refractionPlaneGeometry = new THREE.IcosahedronGeometry(500,4);
		this.refractionPlaneGeometry.mergeVertices();
    	this.refractionPlaneGeometry.computeFaceNormals();
    	this.refractionPlaneGeometry.computeVertexNormals(true);
		this.refractionShader = new RefractionShader();
	 

		// this.refractionPlaneMaterial = new THREE.MeshBasicMaterial({color: 0xffffff, envMap: testCube, side: THREE.DoubleSide});
		// this.refractionPlaneMaterial = new THREE.MeshBasicMaterial({color: 0xffffff, envMap: this.cubeCamera.renderTarget, side: THREE.DoubleSide});
		this.refractionPlaneMaterial = new THREE.ShaderMaterial({
			fragmentShader: this.refractionShader.fragmentShader,
			vertexShader: this.refractionShader.vertexShader,
			uniforms: this.refractionShader.uniforms,
			side: THREE.DoubleSide
		});
		this.refractionPlaneMaterial.uniforms["envMap"].value = this.cubeCamera.renderTarget;
		this.refractionPlaneMaterial.uniforms["alpha"].value = textures[0];
		// this.refractionPlaneMaterial.uniforms["map"].value = THREE.ImageUtils.loadTexture(PATH + "textures/8069-3000x3000-seamless-white-foam-water-texture-image-2-1559.png");
		// this.refractionPlaneMaterial.uniforms["map"].value = textures[0];

		// console.log(testCube);
		// this.refractionPlaneMaterial.uniforms["envMap"] = testCube;
		// this.refractionPlaneMaterial.uniforms["map2"] = null;

		this.refractionPlane = new THREE.Mesh(this.refractionPlaneGeometry, this.refractionPlaneMaterial);
		this.scene.add(this.refractionPlane);
		this.refractionPlane.position.z = 1500;
		// this.refractionPlane.rotation.y += 0.05;

	}

	this.update = function(){

		// for(var i = 0; i < this.texturePlanes.length; i++){
			// this.texturePlanes[i].update();
		// }
		// this.refractionPlaneMaterial.uniforms["envMap"].value = this.cubeCamera.renderTarget;
		this.refractionPlaneMaterial.uniforms["time"].value = time;
		this.refractionPlaneMaterial.uniforms["mouse"].value = mouse;

		// for(var j = 0; j < this.refractionPlaneGeometry.vertices.length; j++){
				// this.refractionPlaneGeometry.vertices[j].z += Math.sin(this.refractionPlaneGeometry.vertices[j].x*0.1 + time)*Math.cos(this.refractionPlaneGeometry.vertices[j].y*0.1 + time)*0.1;
		// }
		this.refractionPlane.geometry.verticesNeedUpdate = true;

		this.refractionPlane.rotation.x = Math.sin(time)*0.05;
		this.refractionPlane.rotation.y = Math.cos(time)*0.05;
	}


}