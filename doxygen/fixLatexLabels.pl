#!/usr/bin/perl

undef $/;
foreach $name (@ARGV) {
    rename $name, "$name.old";
    open(FILE,"$name.old");
    unlink "$name";
    open(NF, "> $name\0") or die "can't open output file!!";
    $line = <FILE>;
    $count = ($line =~ s/\\label{\K..\/...\///gm);
    $count = $count + ($line =~ s/\\hypertarget{\K..\/...\///gm);
    print "replacing $count words in $name. \n";
    print NF $line;
    close FILE;
    unlink "$name.old";
};
