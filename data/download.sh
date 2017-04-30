function rmtemp {
	rm temp.bz2
}
trap rmtemp EXIT

cd data
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O temp.bz2
bzip2 -d temp.bz2