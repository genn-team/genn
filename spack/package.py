# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *
import glob


class Genn(PythonPackage):
    """GeNN is a GPU-enhanced Neuronal Network simulation environment based on
    code generation for Nvidia CUDA."""

    homepage = "https://genn-team.github.io/genn/"
    url      = "https://github.com/genn-team/genn/archive/refs/tags/4.6.0.tar.gz"

    version('4.6.0', sha256='5e5ca94fd3a56b5b963a4911ea1b2130df6fa7dcdde3b025bd8cb85d4c2d3236')

    conflicts('%gcc@:4.9.3')
    depends_on('gmake',      type='build')
    depends_on('googletest', type=('build', 'run', 'test'))
    depends_on('lcov',       type=('build', 'run', 'test'))

    variant('cuda', default=True, description='Enable CUDA support')
    depends_on('cuda', when='+cuda')

    variant('python', default=True, description='Enable PyGeNN')
    extends('python', when='+python')
    depends_on('python@3.8.0:',                when='+python', type=('build', 'run', 'test'))
    depends_on('py-numpy@1.17.0:',             when='+python', type=('build', 'run', 'test'))
    depends_on('py-six',                       when='+python', type=('build', 'run', 'test'))
    depends_on('py-deprecated',                when='+python', type=('build', 'run', 'test'))
    depends_on('py-psutil',                    when='+python', type=('build', 'run', 'test'))
    depends_on('py-importlib-metadata@1.0.0:', when='+python', type=('build', 'run', 'test'))
    depends_on('swig',                         when='+python', type=('build', 'run', 'test'))

    patch('include_path.patch')
    patch('tests_gtest.patch')

    def patch(self):
        files = set(glob.glob("tests/**/Makefile", recursive=True))
        filter_file('\$\(GTEST_DIR\)/src/gtest-all\.cc',
                    '-L$(GTEST_DIR)/lib -lgtest', *files)
        filter_file('\$\(GTEST_DIR\)/src/gtest_main\.cc',
                    '-L$(GTEST_DIR)/lib -lgtest_main', *files)

    def build(self, spec, prefix):
        make('PREFIX={}'.format(prefix), 'install')
        if '+python' in self.spec:
            make('DYNAMIC=1', 'LIBRARY_DIRECTORY={}/pygenn/genn_wrapper/'.format(self.stage.source_path))
            super(Genn, self).build(spec, prefix)

    def install(self, spec, prefix):
        install_tree('bin', prefix.bin)
        install_tree('include', prefix.include)
        mkdirp(prefix.src)
        install_tree('src', prefix.src)
        install_tree('tests', prefix.tests)
        install('version.txt', prefix)
        install('Makefile', prefix)
        if '+python' in self.spec:
            super(Genn, self).install(spec, prefix)

    def setup_run_environment(self, env):
        env.append_path('CPLUS_INCLUDE_PATH', self.prefix.include)
        if '+cuda' in self.spec:
            env.append_path('CUDA_PATH', self.spec['cuda'].prefix)
        if '+python' in self.spec:
            super(Genn, self).setup_run_environment(env)

    def test(self):
        env['CC'] = self.compiler.cc
        env['CXX'] = self.compiler.cxx
        # workaround for https://github.com/spack/spack/issues/20553
        env['GTEST_DIR'] = self.spec['googletest'].prefix
        env['LD_LIBRARY_PATH'] = self.spec['googletest'].prefix.lib
        env['PATH'] = '{}:{}'.format(self.spec['lcov'].prefix.bin, env['PATH'])
        # overriding automatic python testing
        with working_dir(self.prefix.tests):
            # FIXME:
            # * expects a non-zero return code for errors
            # * also too slow for a simple install test?
            # ⇒ provide another simple install test?
            self.run_test('run_tests.sh')
        if '+python' in self.spec:
            super(Genn, self).test()
