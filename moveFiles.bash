## Print LOL Cat if -c argument is provided. 
if [ $2 = "c" ] 
    then  
    echo "LOL CTZ"
    cat test/parserTest/.catASCII.txt
fi

## Test the test cases which should pass:
echo -e '\x1B[35mBeginning pass tests...\e[0m' 

totalP=0
pp=0
pf=0

for file in `ls test/parserTest/pass/*.ic`
    do
    #echo "$file"
    if [ $1 = "v" ] 
        then  
        scala -cp bin:tools/java-cup-11a.jar ic.Compiler $file
        else 
        scala -cp bin:tools/java-cup-11a.jar ic.Compiler $file >/dev/null 2>&1
    fi
    #scala -cp bin ic.Compiler $file >&-
    exitCode=$?
    ((totalP++))
    #echo "exit code: " $exitCode
        if [ $exitCode == 0 ]
            then 
            echo -e "\x1B[32m$file Passed\x1B[0m"
            ((pp++))
            else
            echo -e "\x1B[31m$file Failed\x1B[0m" 
            ((pf++))
        fi
done
echo $pp "tests passed and " $pf " failed, of " $totalP " total pass tests."


## Test the test cases which should fail: 
echo -e "\x1B[35m\nBeginning fail tests...\x1B[0m"

totalf=0
fp=0
ff=0

for file in `ls test/parserTest/fail/*.ic`
    do
    #echo "$file"
    
    if [ $1 = "v" ] 
        then  
        scala -cp bin:tools/java-cup-11a.jar ic.Compiler $file
        else 
        scala -cp bin:tools/java-cup-11a.jar ic.Compiler $file >/dev/null 2>&1
    fi
    
    exitCode=$?
    #echo "Exit code: " $exitCode
    ((totalf++))
        if [ $exitCode == 1 ]
            then 
            echo -e "\x1B[32m$file Failed\x1B[0m"
            ((ff++))
            else
            echo -e "\x1B[31m$file Passed\x1B[0m"
            ((fp++))
        fi
    #echo 
done
echo $ff "tests failed and " $fp " passed, of " $totalf " total fail tests."


#Check how all the tests went and produce a nice summary of results:
echo
echo "SUMMARY: "
echo $pp "tests passed and " $pf " failed, of " $totalP " total pass tests." 
echo $ff "tests failed and " $fp " passed, of " $totalf " total fail tests."

if [ $pp -eq $totalP ] && [ $ff -eq $totalf ] 
    then
    echo "Success."
    else
    echo "Failed."
fi