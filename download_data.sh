if [[ "$1" == "help" ]] || [ "$#" -eq 0 ]; then
    echo "example usage: ./download_data motorbike airplane"
else 
    num_categories=$#;
    ## each category has ~2 GB of data
    total_size="$(($num_categories * 2))"

    while true; do
        read -p "Download "${num_categories}" datasets? ("${total_size}" GB)`echo $'\n> '`" yn
        
        case $yn in
            [Yy]* ) 
                for category in "$@"
                do
                    for data in "test" "val" "train"
                    do
                        ## eg, airplane_test
                        filename=${category}_${data}
                        ## download tar from csail server
                        wget rin.csail.mit.edu/data/${filename}.tar.gz; 
                        echo "Extracting to "dataset/output/${filename}"..."
                        ## extract tar
                        tar xf ${filename}.tar.gz 
                        ## move to data folder
                        mv ${filename} dataset/output/${filename}
                        ## remove tar
                        rm ${filename}.tar.gz
                    done
                done
                break;;
            
            [Nn]* ) 
                exit;;
            
            * ) 
                echo "Please answer yes or no.";;

        esac
    done
fi
