#!/bin/sh
rtdir='/home/data/lwb/code'
src='dynamicbev'
dst='NNAVM'
commit_detail='fev2bev unit test passed'

cd ${rtdir}
mv ${dst} ${dst}_bp
cp -r ${rtdir}/${src} ${rtdir}/${dst}
cd ${dst}
rm -rf .git README.md .gitignore
cd ..
cp -r ${dst}_bp/.git ${dst}/
cp ${dst}_bp/README.md ${dst}/
cp ${dst}_bp/.gitignore ${dst}/
rm -rf ${dst}_bp

for n in ${src} ${dst}
do
    cd ${rtdir}/${n}
    git add .
    git status
    git commit -m \"${commit_detail}\"
    git push
done

cd ${rtdir}/${src}