#include <cstddef> // NULL
#include <iostream>
#include <iomanip>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/version.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
namespace mlut
{
    class xy
    {
        friend std::ostream & operator<<(std::ostream &os, const xy &xy);
        friend class boost::serialization::access;
        
        int x;
        int y;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & x;
            ar & y;
        }
        public:
            xy(){};
            xy(int x, int y):x(x),y(y){};
            int getX(){return x;}
            int getY(){return y;}
    };
    std::ostream& operator<< (std::ostream &os, const xy &xy)
    {
        return os << " xy : " << xy.x << ", " << xy.y << "\n";
    }

    class txty
    {   
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const txty &txty);
        
        int tx, ty;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & tx;
            ar & ty;
        }
        public:
            txty(){};
            txty(int tx, int ty):tx(tx), ty(ty){};
            int getTX(){return tx;}
            int getTY(){return ty;}
    };

    BOOST_SERIALIZATION_ASSUME_ABSTRACT(txty)
    std::ostream& operator<< (std::ostream &os, const txty &txty)
    {
        return os << " txty : " << txty.tx << ", " << txty.ty << "\n";
    }

    class ipCoefs
    {   
        friend class boost::serialization::access;
        friend std::ostream & operator<<(std::ostream &os, const ipCoefs &coefs);
        
        double bl, tl, br, tr;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & bl; 
            ar & tl;
            ar & br; 
            ar & tr;
        }
        public:
            ipCoefs(double bl, double tl, double br, double tr):bl(bl), tl(tl), br(br), tr(tr){};

            ipCoefs(){};

            double getBL(){return bl;}
            double getTL(){return tl;}
            double getBR(){return br;}
            double getTR(){return tr;}
    };
    
    BOOST_SERIALIZATION_ASSUME_ABSTRACT(ipCoefs)
    std::ostream& operator<< (std::ostream &os, const ipCoefs &coefs)
    {
        return os << " coefs : " << coefs.bl << ", " << coefs.tl << ", " << coefs.br << ", " << coefs.tr << "\n";
    }

    class mappingData
    {       
        friend std::ostream & operator<<(std::ostream &os, const mappingData &md);
        friend class boost::serialization::access;
        
        xy _xy;
        txty _txty;
        ipCoefs _coefs;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar.register_type(static_cast<xy *>(NULL));
            ar.register_type(static_cast<txty *>(NULL));
            ar.register_type(static_cast<ipCoefs *>(NULL));

            ar & _xy;
            ar & _txty;
            ar & _coefs;
        }
        public:
            mappingData(){};
            mappingData(const int & x, const int & y, const int & tx, const int & ty, const double & bl, const double & tl, const double & br, const double & tr)
                        :_xy(x,y), _txty(tx,ty), _coefs(bl,tl,br,tr){};
            xy getXY(){ return _xy;}
            txty getTXTY(){ return _txty;}
            ipCoefs getIpCoefs(){ return _coefs;}
    };

    std::ostream& operator<< (std::ostream &os, const mappingData &md)
    {
        return os << " " << md._xy << " , " << md._txty << " , " << md._coefs << "\n";
    }

    class mLUT
    {
        friend std::ostream & operator<<(std::ostream &os, const mLUT &lut);
        friend class boost::serialization::access;
        
        typedef mappingData * md_ptr;
        std::vector<md_ptr> maps;
        
        template<class Archive>

        void serialize(Archive& ar, const unsigned int version)
        {
            ar.register_type(static_cast<mappingData *>(NULL));

            ar & maps;
        }
        public:
            mLUT(){};
            void append(mappingData * _md)
            {
                maps.insert(maps.end(), _md);
            }

            std::vector<md_ptr> getMaps(){return maps;}
    };

    std::ostream & operator<<(std::ostream &os, const mLUT &lut)
    {
        std::vector<mappingData *>::const_iterator it;
        for(it = lut.maps.begin(); it != lut.maps.end(); it++){
            os << '\n' << std::hex << "0x" << *it << std::dec << ' ' << **it;
        }
        return os;
    }

    void save(const mLUT &lut, const char * filename){
        // make an archive
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oa(ofs);
        oa << lut;
    }

    void load(mLUT &s, const char* filename)
    {
        // open the archive
        std::ifstream ifs(filename);
        boost::archive::binary_iarchive ia(ifs);

        // restore the schedule from the archive
        ia >> s;
    }
}