# comp2
sudo apt-get install git
git-config --global user.email "your@email.here"
git-config --global user.name "your_name_here"
make a directory you want to hold the repository folder
  where you want to save the files from the repository
inside new directory: git clone https://github.com/jsimms22/jsimmons
commands that are useful
  git status -> show the working tree status
  git add file.test -> add file contents to the index
  git commit -m "add comments here" -> record changes to the repository
  git push -> update remote refs along with associated objects
  git -> will list all github terminal commands
  
https://www.youtube.com/watch?v=E8TXME3bzNs
30 minute tutorial on github

https://portal.xsede.org/psc-bridges#overview
system overview for bridges cluster

http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1
dgemm from blas source code

https://ac.els-cdn.com/S1877050912001354/1-s2.0-S1877050912001354-main.pdf?_tid=44415eac-0ae9-11e8-8be7-00000aab0f26&acdnat=1517885942_fdc507aecc94f0dadde2746529b3918e
paper discussing comparisons in dgemm algorithms, might be useful for our considerations

random sites that have read:
1) https://wiki.gentoo.org/wiki/GCC_optimization
2) https://stackoverflow.com/questions/20367246/loop-tiling-how-to-choose-block-size
3) https://www.psc.edu/bridges/user-guide/system-configuration
4) https://people.eecs.berkeley.edu/~demmel/cs267_Spr99/Lectures/Lect_02_1999b.pdf
5) https://www2.eecs.berkeley.edu/Pubs/TechRpts/1998/CSD-98-1020.pdf
6) https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec03.pdf
7) http://web.cs.ucdavis.edu/~bai/ECS231/optmatmul.pdf
8) http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html#gaeda3cbd99c8fb834a60a6412878226e1

resources for AVX instruction and intrinsics:
1) https://pdfs.semanticscholar.org/852c/0115d6011b6cd2746d18f56d64a53e65af5d.pdf
2) https://software.intel.com/en-us/articles/how-to-use-intrinsics
3) https://github.com/changlan/dgemm/blob/master/src/dgemm-blocked.c
