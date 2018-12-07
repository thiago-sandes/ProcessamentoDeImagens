import welerson_augusto as w
def testeConvolve(file):
    img = w.imreadgray(file)
    k = [[-1, -1, -1], [-1, 8, -1], [-1,-1, -1]]
    ans = w.convolve(img, k)
    w.imshow(ans)
    return ans
    
def testeBlur(file):
	img = w.imreadgray(file)
	w.imshow(img)
	ans = w.blur(img)
	w.imshow(ans)

