# TODO: can the all and debug be unified somehow?
MAKEFLAGS += --silent

project_name:=heattests
root_dir:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
source_dir:=$(root_dir)
install_dir:=$(root_dir)
build_dir_prefix:=$(root_dir)/build

.PHONY: all test clean debug

.ONESHELL:
all:
	build_type=Release
	build_dir=$(build_dir_prefix)/$${build_type}
	cmake \
		-B $${build_dir} \
		-S $(source_dir) \
		-DCMAKE_BUILD_TYPE:STRING=$${build_type} \
		-DCMAKE_INSTALL_PREFIX:PATH=$(install_dir) \
		-Dproject_name=$(project_name) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cmake \
		--build $${build_dir} \
		--target install -j8
	cp $${build_dir}/compile_commands.json $(source_dir)

clean:
	rm -rf $(build_dir_prefix) $(project_name)

test: all
	ctest --test-dir $(build_dir_prefix)/Release

.ONESHELL:
debug:
	build_type=Debug
	build_dir=$(build_dir_prefix)/$${build_type}
	cmake \
		-B $${build_dir} \
		-S $(source_dir) \
		-DCMAKE_BUILD_TYPE:STRING=$${build_type} \
		-Dproject_name=$(project_name) \
		-Dpackage_tests=OFF

	cmake \
		--build $${build_dir} \
		--target all -j8
