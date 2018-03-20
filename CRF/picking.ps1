#picking training annotations
# $RootPath = '.\cls\'
# $FileName = (Get-Content '.\train.txt').Split('\n')
# foreach($file in $FileName){
#     $FilePath = $RootPath + $file + '.mat'
#     Copy-Item $FilePath '.\train'
# }

#picking validation annotations
# $RootPath = '.\cls\'
# $FileName = (Get-Content '.\val.txt').Split('\n')
# foreach($file in $FileName){
#     $FilePath = $RootPath + $file + '.mat'
#     Copy-Item $FilePath '.\val'
# }

#picking training images
# $RootPath = '.\img\'
# $FileName = (Get-Content '.\train.txt').Split('\n')
# foreach($file in $FileName){
#     $FilePath = $RootPath + $file + '.jpg'
#     Copy-Item $FilePath '.\images\training'
# }

#picking validation images
$RootPath = '.'
$FileName = (Get-Content '.\val_ori.txt')
foreach($file in $FileName){
    $FilePath = $RootPath + $file 
    Copy-Item $FilePath '.\1'
}