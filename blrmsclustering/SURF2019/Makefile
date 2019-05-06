SHELL = bash

PREFIX=env

clean:
	rm -rf ${PREFIX}

.ONESHELL:
install: clean
	$(eval TMP := $(shell mktemp -d))	
	pushd ${TMP}
	curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
	chmod +x Miniconda3-latest-Linux-x86_64.sh
	popd
	${TMP}/Miniconda3-latest-Linux-x86_64.sh -b -f -p ${PREFIX}
	source ${PREFIX}/etc/profile.d/conda.sh; conda activate
	export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}.
	bash -c 'IFS=\#; for i in $$(cat PKGS); do bash -c "conda install -y $$(echo $$i | sed "s/^ /-c /g" | tr "\n" " ")"; done'
enter:
	source ${PREFIX}/etc/profile.d/conda.sh; conda activate
	export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}
	${SHELL}
