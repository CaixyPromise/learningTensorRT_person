#ffserver -f server1.conf &
#ffserver -f server2.conf &
#ffserver -f server3.conf &
#ffserver -f server4.conf &
#ffmpeg -i 1.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1234/feed1.ffm &
#ffmpeg -i 2.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1235/feed2.ffm &
#ffmpeg -i 3.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1236/feed3.ffm &
#ffmpeg -i 4.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1237/feed4.ffm &
./rtsp-simple-server rtsp-simple-server.yml &
# ffmpeg -re -stream_loop -1 -i c3_1080.mp4 -vcodec copy -acodec copy -b:v 8M -f rtsp -rtsp_transport tcp rtmp://localhost:6006/live.sdp &
# rtmp
# ffmpeg -re -stream_loop -1 -i c3_1080.mp4 -c:v libx264 -preset veryfast -b:v 8M -c:a aac -b:a 128k -f flv rtmp://localhost:6006/live/stream