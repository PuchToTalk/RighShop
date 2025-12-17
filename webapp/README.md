# RighShop Demo with Local Webapp
Demo setup with local machine (Raspberry Pi) and Set-top box <br>
<br>

1. ssh  innovo@<IP_address> (check ifconfig before)
<br>
2. git clone righshop-demo
<br>
3. cd righshop-demo/webapp
<br>
4. python -m venv snap-env <br>
   source snap-env/bin/activate 
   <br>
5. pip install -r requirements.txt
<br>

6. nano ~/.bashrc
<br>
Add at the last line: <br>
```
if [ "$(tty)" = "/dev/tty1" ]; then
        ifconfig
        cd /home/innovo/righshop-demo
        git stash
        git pull origin
        cd webapp/
        sudo /home/innovo/righshop-demo/webapp/snap-env/bin/python /home/innovo/righshop-demo/webapp/detect_device.py
        sudo /home/innovo/righshop-demo/webapp/snap-env/bin/python /home/innovo/righshop-demo/webapp/run.py
fi
```
<br>

7. crontab -e
<br>
Add at the last line:<br>
@reboot /home/innovo/righshop-demo/webapp/snap-env/bin/python /home/innovo/righshop-demo/webapp/app.py
<br>

8. reboot (to see changes)<br>
<br>
Setup hardware:<br>
-Connect set-top box to local network and monitor<br>
<br>
Configure adb:<br>
sudo apt-get install adb
<br>
/usr/bin/adb shell
<br>
mkdir data/temp
<br>

<br>
Set-top box firmware setup:<br>
-refer to: https://righ-wiki.atlassian.net/wiki/spaces/~712020bc93b8e30954463785935dae23b22466/pages/25821188/Android+Setup+Box<br>
<br>
Google Cloud setup:<br>
https://righ-wiki.atlassian.net/wiki/spaces/~712020bc93b8e30954463785935dae23b22466/pages/68681729/Dogfood+Server+for+RighShop<br>
<br>
```
Webapp Directory Structure
webapp/
├── app.py
├── templates/
│   └── index.html
└── static/
    ├── 1.jpg
    └── output.txt
```
<br>

 Access Webapp: <IP_address>:5001<br>

 
