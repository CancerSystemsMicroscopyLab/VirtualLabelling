
// select folder
dir = getDirectory("Choose a Directory ");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
	if(endsWith(list[i], ".tif")){
		img_name = replace(list[i], '\.tif', '');
		subtract_median_background(img_name);
	}
}


function subtract_median_background(img_name) {
	print(dir + img_name + ".tif");
	open(dir + img_name + ".tif");
	selectWindow(img_name+".tif");
	run("Median...", "radius=60");
//	run("Gaussian Blur...", "sigma=5");
	open(dir + img_name + ".tif");
	imageCalculator("Subtract create 32-bit", img_name+"-1.tif", img_name+".tif");
	run("Enhance Contrast", "saturated=0.2");
	run("Conversions...", "scale");
	run("8-bit");
	saveAs("Tiff", dir+"/results/"+img_name+".tif");
	close();
	close();
	close();
}
