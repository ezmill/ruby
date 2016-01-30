function TexturePlane(SCENE, CAMERA, RENDERER, TEXTURE, SPEED){
	this.scene = SCENE;
	this.camera = CAMERA;
	this.renderer = RENDERER;
	this.texture = TEXTURE;
	this.speed = SPEED;

	this.init = function(){
		// this.geometry = new THREE.PlaneGeometry(this.texture.image.width/10, this.texture.image.height/10);
		this.geometry = new THREE.PlaneGeometry(this.texture.image.width, this.texture.image.height);
		// this.geometry = new THREE.PlaneGeometry(renderSize.x, renderSize.y);
		this.material = new THREE.MeshBasicMaterial({
			map: this.texture,
			side: THREE.DoubleSide
		})
		this.mesh = new THREE.Mesh(this.geometry, this.material);
		this.scene.add(this.mesh);
		// this.mesh.position.x = Math.random()*(renderSize.x - renderSize.x/2) - this.texture.image.width/5;
		// this.mesh.position.y = Math.random()*renderSize.y*4 - renderSize.y*2;
		// this.mesh.position.y = Math.random()*-renderSize.y*2;
		// this.mesh.position.y = -renderSize.y/2;
		// this.mesh.position.z = Math.random()*1000;
	}

	this.update = function(){
		// this.mesh.position.y += 1.0;
	}
}